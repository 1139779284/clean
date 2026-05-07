from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    discover_external_attack_datasets,
    infer_attack_goal,
    run_external_hard_suite_for_yolo,
    write_external_hard_suite_outputs,
)
from model_security_gate.detox.strong_train import StrongDetoxConfig, run_strong_detox_training
from model_security_gate.detox.yolo_dataset import image_to_label_path, parse_yolo_data_yaml
from model_security_gate.utils.io import read_yaml, write_json, write_yaml


@dataclass
class TargetedRepairConfig:
    model: str
    data_yaml: str
    out_dir: str
    external_roots: Sequence[str] = field(default_factory=tuple)
    target_classes: Sequence[str | int] = field(default_factory=tuple)
    attack_names: Sequence[str] = field(default_factory=tuple)
    repair_goal: str = "oda"
    failure_rows_csv: str | None = None
    teacher_model: str | None = None
    device: str | None = None
    imgsz: int = 416
    conf: float = 0.25
    batch: int = 8
    epochs: int = 3
    lr: float = 1e-5
    max_images_per_attack: int = 20
    replay_max_images_per_attack: int = 20
    failure_repeat: int = 12
    oda_full_image_extra_repeat: int = 0
    oda_focus_crops: bool = False
    oda_focus_crop_repeat: int = 2
    clean_anchor_images: int = 24
    clean_anchor_seed: int = 42
    max_single_attack_worsen: float = 0.02
    max_allowed_external_asr: float = 0.10
    max_train_images: int | None = None
    max_val_images: int | None = None
    # Loss weights are intentionally ODA/OGA-specific and conservative.
    lambda_task: float = 0.30
    lambda_oda_recall: float = 0.80
    lambda_oda_matched: float = 2.00
    lambda_oga_negative: float = 0.00
    lambda_output_distill: float = 0.0
    lambda_feature_distill: float = 0.0
    lambda_nad: float = 0.0
    lambda_attention: float = 0.0
    lambda_adv: float = 0.0
    weight_decay: float = 5e-4
    save_every: int = 1


def load_failure_rows_csv(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _target_ids_from_names(data_yaml: str | Path, target_classes: Sequence[str | int]) -> list[int]:
    names = read_yaml(data_yaml).get("names", {})
    if isinstance(names, list):
        idx_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        idx_to_name = {int(k): str(v) for k, v in dict(names).items()}
    inv = {v.lower(): k for k, v in idx_to_name.items()}
    out: list[int] = []
    for item in target_classes:
        text = str(item)
        if text.isdigit():
            out.append(int(text))
        elif text.lower() in inv:
            out.append(int(inv[text.lower()]))
        else:
            raise ValueError(f"Unknown target class {item!r}; available={idx_to_name}")
    return sorted(set(out))


def _select_attack_names(
    discovered_names: Sequence[str],
    requested: Sequence[str] | None,
    repair_goal: str,
) -> list[str]:
    if requested:
        return [str(x) for x in requested]
    goal = str(repair_goal).lower()
    if goal in {"all", "mixed"}:
        return list(discovered_names)
    return [name for name in discovered_names if infer_attack_goal(name) == goal]


def _copy_clean_anchors(output_dataset_dir: str | Path, source_data_yaml: str | Path, max_images: int, seed: int = 42) -> dict[str, Any]:
    if max_images <= 0:
        return {"added": 0}
    info = parse_yolo_data_yaml(source_data_yaml)
    paths = list(info.train_images or info.val_images)
    if not paths:
        return {"added": 0, "warning": "no_clean_anchor_source_images"}
    rng = np.random.default_rng(int(seed))
    if len(paths) > int(max_images):
        idx = rng.choice(len(paths), size=int(max_images), replace=False)
        paths = [paths[int(i)] for i in sorted(idx.tolist())]
    out_dir = Path(output_dataset_dir)
    img_out = out_dir / "images" / "train"
    lab_out = out_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    added = 0
    for i, path in enumerate(paths):
        stem = f"clean_anchor_{i:05d}_{path.stem}".replace(" ", "_")
        dest_img = img_out / f"{stem}{path.suffix.lower() if path.suffix else '.jpg'}"
        dest_lab = lab_out / f"{stem}.txt"
        shutil.copy2(path, dest_img)
        label_path = image_to_label_path(path)
        if label_path.exists():
            shutil.copy2(label_path, dest_lab)
        else:
            dest_lab.write_text("", encoding="utf-8")
        added += 1
    return {"added": added}


def _write_repair_data_yaml(output_dataset_dir: str | Path, source_data_yaml: str | Path) -> Path:
    out_dir = Path(output_dataset_dir)
    source = read_yaml(source_data_yaml)
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": source.get("names", {}),
    }
    out = out_dir / "data.yaml"
    write_yaml(out, data)
    return out


def _matrix_from_external(result: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in ((result.get("summary") or {}).get("asr_matrix") or {}).items()}


def _blocked_by_worsening(candidate: Mapping[str, Any], baseline: Mapping[str, Any], max_worsen: float) -> list[str]:
    cand = _matrix_from_external(candidate)
    base = _matrix_from_external(baseline)
    blocked: list[str] = []
    for key, cand_value in cand.items():
        if key in base and cand_value - base[key] > float(max_worsen):
            blocked.append(key)
    return blocked


def _external_score(result: Mapping[str, Any]) -> float:
    summary = result.get("summary") or {}
    max_asr = float(summary.get("max_asr", 1.0))
    mean_asr = float(summary.get("mean_asr", max_asr))
    return max_asr + 0.35 * mean_asr


def select_final_repair_candidate(candidate_rows: Sequence[Mapping[str, Any]], fallback_model: str) -> dict[str, Any]:
    """Select only unblocked candidates; otherwise explicitly roll back.

    ``best_by_score`` is retained for diagnosis, but a blocked candidate must
    never become the final model because that would silently publish a known
    ASR regression.
    """
    rows = [dict(row) for row in candidate_rows]
    best_by_score = min(rows, key=lambda r: float(r["score"])) if rows else None
    eligible = [row for row in rows if not row.get("blocked_attacks")]
    best = min(eligible, key=lambda r: float(r["score"])) if eligible else None
    return {
        "final_model": str(best["model"]) if best else str(fallback_model),
        "best": best,
        "best_by_score": best_by_score,
        "rolled_back": best is None,
    }


def run_targeted_repair(cfg: TargetedRepairConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "targeted_repair_config.json", asdict(cfg))

    target_ids = _target_ids_from_names(cfg.data_yaml, cfg.target_classes)
    attack_datasets = discover_external_attack_datasets(cfg.external_roots)
    selected_attacks = _select_attack_names([ds.name for ds in attack_datasets], cfg.attack_names, cfg.repair_goal)

    before_cfg = ExternalHardSuiteConfig(
        roots=tuple(cfg.external_roots),
        conf=float(cfg.conf),
        imgsz=int(cfg.imgsz),
        max_images_per_attack=int(cfg.max_images_per_attack),
        seed=42,
    )
    before = run_external_hard_suite_for_yolo(
        cfg.model,
        data_yaml=cfg.data_yaml,
        target_classes=cfg.target_classes,
        cfg=before_cfg,
        device=cfg.device,
    )
    before_json, before_rows_csv = write_external_hard_suite_outputs(before, out_dir / "eval_before_external")
    failure_rows = load_failure_rows_csv(cfg.failure_rows_csv) or list(before.get("rows") or [])
    failure_rows = [row for row in failure_rows if _truthy(row.get("success"))]

    dataset_dir = out_dir / "01_failure_repair_dataset"
    replay_stats = append_external_replay_samples(
        output_dataset_dir=dataset_dir,
        attack_datasets=attack_datasets,
        target_class_ids=target_ids,
        selected_attack_names=selected_attacks,
        max_images_per_attack=int(cfg.replay_max_images_per_attack),
        failure_rows=failure_rows,
        failure_only=True,
        repeat=int(cfg.failure_repeat),
        oda_full_image_extra_repeat=int(cfg.oda_full_image_extra_repeat),
        oda_focus_crops=bool(cfg.oda_focus_crops),
        oda_focus_crop_repeat=int(cfg.oda_focus_crop_repeat),
    )
    clean_stats = _copy_clean_anchors(dataset_dir, cfg.data_yaml, int(cfg.clean_anchor_images), seed=int(cfg.clean_anchor_seed))
    repair_data_yaml = _write_repair_data_yaml(dataset_dir, cfg.data_yaml)

    goal = str(cfg.repair_goal).lower()
    use_oda = goal in {"oda", "mixed", "all"}
    use_oga = goal in {"oga", "mixed", "all"}
    train_cfg = StrongDetoxConfig(
        model=cfg.model,
        data_yaml=str(repair_data_yaml),
        out_dir=str(out_dir / "02_repair_train"),
        teacher_model=cfg.teacher_model,
        epochs=int(cfg.epochs),
        batch=int(cfg.batch),
        imgsz=int(cfg.imgsz),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        num_workers=0,
        device=cfg.device,
        max_train_images=cfg.max_train_images,
        max_val_images=cfg.max_val_images,
        target_class_ids=target_ids,
        lambda_task=float(cfg.lambda_task),
        lambda_adv=float(cfg.lambda_adv),
        lambda_output_distill=float(cfg.lambda_output_distill if cfg.teacher_model else 0.0),
        lambda_feature_distill=float(cfg.lambda_feature_distill if cfg.teacher_model else 0.0),
        lambda_nad=float(cfg.lambda_nad if cfg.teacher_model else 0.0),
        lambda_attention=float(cfg.lambda_attention),
        lambda_prototype=0.0,
        lambda_proto_suppress=0.0,
        lambda_oda_recall=float(cfg.lambda_oda_recall if use_oda else 0.0),
        lambda_oda_matched=float(cfg.lambda_oda_matched if use_oda else 0.0),
        lambda_oga_negative=float(cfg.lambda_oga_negative if use_oga else 0.0),
        use_teacher=bool(cfg.teacher_model),
        use_prototype=False,
        use_attention=bool(cfg.lambda_attention > 0),
        save_every=int(cfg.save_every),
    )
    train_report = run_strong_detox_training(train_cfg)

    candidates: list[Path] = []
    train_dir = Path(train_cfg.out_dir)
    for key in ("best_model", "final_model"):
        value = train_report.get(key)
        if value:
            candidates.append(Path(str(value)))
    candidates.extend(sorted(train_dir.glob("epoch_*.pt")))
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if path.exists() and key not in seen:
            unique_candidates.append(path)
            seen.add(key)

    candidate_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    for idx, model_path in enumerate(unique_candidates, start=1):
        result = run_external_hard_suite_for_yolo(
            str(model_path),
            data_yaml=cfg.data_yaml,
            target_classes=cfg.target_classes,
            cfg=before_cfg,
            device=cfg.device,
        )
        eval_dir = out_dir / "03_candidate_external" / f"c{idx:02d}_{model_path.stem}"
        cand_json, cand_rows_csv = write_external_hard_suite_outputs(result, eval_dir)
        summary = result.get("summary") or {}
        blocked = _blocked_by_worsening(result, before, float(cfg.max_single_attack_worsen))
        row = {
            "model": str(model_path),
            "external_json": str(cand_json),
            "external_rows_csv": str(cand_rows_csv),
            "external_max_asr": summary.get("max_asr"),
            "external_mean_asr": summary.get("mean_asr"),
            "score": _external_score(result),
            "blocked_attacks": blocked,
            "accepted": (not blocked) and float(summary.get("max_asr", 1.0)) <= float(cfg.max_allowed_external_asr),
        }
        candidate_rows.append(row)
        if not blocked:
            if best_row is None or float(row["score"]) < float(best_row["score"]):
                best_row = row
                best_result = result

    selection = select_final_repair_candidate(candidate_rows, cfg.model)
    final_row = selection["best"]
    final_model = selection["final_model"]
    manifest = {
        "status": "passed" if final_row and final_row.get("accepted") else "failed_external_asr_or_worsening",
        "final_model": final_model,
        "rolled_back": bool(selection["rolled_back"]),
        "target_class_ids": target_ids,
        "selected_attacks": selected_attacks,
        "before_external_json": str(before_json),
        "before_rows_csv": str(before_rows_csv),
        "before_summary": before.get("summary"),
        "repair_data_yaml": str(repair_data_yaml),
        "replay_stats": replay_stats,
        "clean_anchor_stats": clean_stats,
        "train_report": train_report,
        "candidate_rows": candidate_rows,
        "best": final_row,
        "best_by_score": selection["best_by_score"],
        "best_summary": best_result.get("summary") if best_result else None,
    }
    write_json(out_dir / "targeted_repair_manifest.json", manifest)
    return manifest
