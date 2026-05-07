from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    discover_external_attack_datasets,
    infer_attack_goal,
    run_external_hard_suite_for_yolo,
    write_external_hard_suite_outputs,
)
from model_security_gate.detox.losses import raw_prediction, supervised_yolo_loss
from model_security_gate.detox.oda_loss_v2 import (
    matched_candidate_oda_loss,
    negative_target_candidate_suppression_loss,
)
from model_security_gate.detox.strong_train import (
    _torch_model,  # existing internal helper; kept here to avoid reimplementing Ultralytics loading edge cases
    load_ultralytics_yolo,
    save_ultralytics_yolo,
)
from model_security_gate.detox.yolo_dataset import (
    image_to_label_path,
    make_yolo_dataloader,
    move_batch_to_device,
    parse_yolo_data_yaml,
)
from model_security_gate.utils.io import read_yaml, write_json, write_yaml


@dataclass
class ODAPostNMSRepairConfig:
    """Surgical ODA repair that optimizes full-image localized recall.

    This is intentionally narrower than Hybrid-PURIFY. It is meant for the
    current residual failure mode where a Pareto/Hybrid candidate already has
    low mean ASR but a small set of full-image ODA failures still has no
    localized GT recall after the external hard-suite evaluator.

    The repair dataset is built from current external ``success=true`` rows and
    preserves full images by default. Crops are not the primary signal because
    they can remove the global trigger/context that caused disappearance.
    """

    model: str
    data_yaml: str
    out_dir: str
    external_roots: Sequence[str] = field(default_factory=tuple)
    target_classes: Sequence[str | int] = field(default_factory=tuple)
    attack_names: Sequence[str] = field(default_factory=tuple)
    failure_rows_csv: str | None = None
    teacher_model: str | None = None
    device: str | None = None

    imgsz: int = 416
    conf: float = 0.25
    batch: int = 4
    epochs: int = 10
    lr: float = 2e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    amp: bool = False
    seed: int = 42

    max_images_per_attack: int = 20
    replay_max_images_per_attack: int = 20
    failure_repeat: int = 24
    clean_anchor_images: int = 8
    clean_anchor_seed: int = 42
    save_every: int = 1

    # The surgical objective: ODA localized candidate should become a final-like
    # strong candidate. Task loss is deliberately small to avoid drifting the
    # already-good Pareto checkpoint back toward ordinary fine-tuning.
    lambda_task: float = 0.03
    lambda_oda_matched: float = 4.0
    lambda_oda_recall: float = 0.0  # old confidence-floor loss is not used here
    lambda_oga_negative: float = 0.0

    # Matched-candidate details. These are sharper than the generic ODA-v2
    # defaults because the repair only sees full-image failed ODA rows.
    oda_iou_threshold: float = 0.03
    oda_center_radius: float = 2.0
    oda_topk: int = 48
    oda_cls_weight: float = 1.4
    oda_box_weight: float = 0.45
    oda_teacher_score_weight: float = 0.25
    oda_teacher_box_weight: float = 0.20
    oda_min_score: float = 0.60
    oda_best_score_weight: float = 1.25
    oda_best_box_weight: float = 0.55
    oda_localized_margin: float = 0.20
    oda_localized_margin_weight: float = 0.90
    negative_other_cls_weight: float = 0.02

    # Candidate selection / rollback.
    max_single_attack_worsen: float = 0.02
    max_allowed_external_asr: float = 0.10
    min_external_score_improvement: float = 1e-6
    require_improvement_for_final: bool = True


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _target_ids_from_names(data_yaml: str | Path, target_classes: Sequence[str | int]) -> list[int]:
    raw = read_yaml(data_yaml).get("names", {})
    if isinstance(raw, list):
        names = {i: str(v) for i, v in enumerate(raw)}
    else:
        names = {int(k): str(v) for k, v in dict(raw).items()}
    inv = {v.lower(): k for k, v in names.items()}
    ids: list[int] = []
    for item in target_classes:
        text = str(item)
        if text.isdigit():
            ids.append(int(text))
        elif text.lower() in inv:
            ids.append(int(inv[text.lower()]))
        else:
            raise ValueError(f"Unknown target class {item!r}; available={names}")
    return sorted(set(ids))


def _select_attack_names(discovered_names: Sequence[str], requested: Sequence[str] | None, goal: str = "oda") -> list[str]:
    if requested:
        return [str(x) for x in requested]
    goal_low = str(goal).lower()
    return [name for name in discovered_names if infer_attack_goal(name) == goal_low]


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
        stem = f"clean_anchor_{i:05d}_{Path(path).stem}".replace(" ", "_")
        dest_img = img_out / f"{stem}{Path(path).suffix.lower() if Path(path).suffix else '.jpg'}"
        dest_lab = lab_out / f"{stem}.txt"
        shutil.copy2(path, dest_img)
        label_path = image_to_label_path(Path(path))
        if label_path.exists():
            shutil.copy2(label_path, dest_lab)
        else:
            dest_lab.write_text("", encoding="utf-8")
        added += 1
    return {"added": added}


def _write_repair_data_yaml(output_dataset_dir: str | Path, source_data_yaml: str | Path) -> Path:
    out_dir = Path(output_dataset_dir)
    raw = read_yaml(source_data_yaml)
    out = out_dir / "data.yaml"
    write_yaml(
        out,
        {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/train",
            "names": raw.get("names", {}),
            "label_mode": "oda_postnms_failure_repair",
        },
    )
    return out


def _matrix(result: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in ((result.get("summary") or {}).get("asr_matrix") or {}).items()}


def _external_score(result: Mapping[str, Any]) -> float:
    s = result.get("summary") or {}
    max_asr = float(s.get("max_asr", 1.0))
    mean_asr = float(s.get("mean_asr", max_asr))
    return max_asr + 0.35 * mean_asr


def _blocked_by_worsening(candidate: Mapping[str, Any], baseline: Mapping[str, Any], max_worsen: float) -> list[str]:
    cand = _matrix(candidate)
    base = _matrix(baseline)
    bad: list[str] = []
    for key, after in cand.items():
        before = base.get(key)
        if before is not None and float(after) - float(before) > float(max_worsen):
            bad.append(key)
    return bad


def _filter_success_rows(rows: Sequence[Mapping[str, Any]], attack_names: Sequence[str]) -> list[dict[str, Any]]:
    wanted = {str(a).lower() for a in attack_names}
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _truthy(row.get("success")):
            continue
        if wanted and str(row.get("attack", "")).lower() not in wanted:
            continue
        out.append(dict(row))
    return out


def _device_from_string(value: str | None) -> torch.device:
    if value:
        if str(value).isdigit():
            return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
        return torch.device(str(value))
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_postnms_candidate(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_score: float,
    fallback_model: str,
    min_improvement: float = 1e-6,
    require_improvement: bool = True,
) -> dict[str, Any]:
    rows = [dict(r) for r in candidate_rows]
    best_by_score = min(rows, key=lambda r: float(r["score"])) if rows else None
    eligible = [r for r in rows if not r.get("blocked_attacks")]
    if require_improvement:
        eligible = [r for r in eligible if float(r["score"]) < float(baseline_score) - float(min_improvement)]
    best = min(eligible, key=lambda r: float(r["score"])) if eligible else None
    return {
        "final_model": str(best["model"]) if best else str(fallback_model),
        "best": best,
        "best_by_score": best_by_score,
        "rolled_back": best is None,
    }


def _build_failure_dataset(
    cfg: ODAPostNMSRepairConfig,
    out_dir: Path,
    target_ids: Sequence[int],
    attack_names: Sequence[str],
    before_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    attack_datasets = discover_external_attack_datasets(cfg.external_roots)
    failure_rows = _read_csv_rows(cfg.failure_rows_csv) or list(before_rows)
    failure_rows = _filter_success_rows(failure_rows, attack_names)
    dataset_dir = out_dir / "01_postnms_failure_dataset"
    replay_stats = append_external_replay_samples(
        output_dataset_dir=dataset_dir,
        attack_datasets=attack_datasets,
        target_class_ids=target_ids,
        selected_attack_names=attack_names,
        max_images_per_attack=int(cfg.replay_max_images_per_attack),
        split="train",
        seed=int(cfg.seed),
        failure_rows=failure_rows,
        failure_only=True,
        repeat=int(cfg.failure_repeat),
        # Do not enable crops here: this stage is explicitly full-image repair.
        oda_focus_crops=False,
    )
    clean_stats = _copy_clean_anchors(dataset_dir, cfg.data_yaml, int(cfg.clean_anchor_images), seed=int(cfg.clean_anchor_seed))
    repair_yaml = _write_repair_data_yaml(dataset_dir, cfg.data_yaml)
    return repair_yaml, replay_stats, clean_stats, failure_rows


def run_oda_postnms_repair(cfg: ODAPostNMSRepairConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "oda_postnms_repair_config.json", asdict(cfg))

    target_ids = _target_ids_from_names(cfg.data_yaml, cfg.target_classes)
    discovered = discover_external_attack_datasets(cfg.external_roots)
    selected_attacks = _select_attack_names([d.name for d in discovered], cfg.attack_names, goal="oda")
    if not selected_attacks:
        raise ValueError("No ODA attacks selected/discovered; pass --attack-names badnet_oda or valid external roots")

    eval_cfg = ExternalHardSuiteConfig(
        roots=tuple(cfg.external_roots),
        conf=float(cfg.conf),
        imgsz=int(cfg.imgsz),
        max_images_per_attack=int(cfg.max_images_per_attack),
        seed=int(cfg.seed),
    )
    before = run_external_hard_suite_for_yolo(
        cfg.model,
        data_yaml=cfg.data_yaml,
        target_classes=cfg.target_classes,
        cfg=eval_cfg,
        device=cfg.device,
    )
    before_json, before_csv = write_external_hard_suite_outputs(before, out_dir / "eval_00_before_external")
    baseline_score = _external_score(before)

    repair_yaml, replay_stats, clean_stats, failure_rows = _build_failure_dataset(
        cfg, out_dir, target_ids, selected_attacks, before.get("rows") or []
    )

    device = _device_from_string(cfg.device)
    yolo = load_ultralytics_yolo(cfg.model, device)
    student = _torch_model(yolo).to(device)
    student.train()

    teacher = None
    if cfg.teacher_model:
        teacher_yolo = load_ultralytics_yolo(cfg.teacher_model, device)
        teacher = _torch_model(teacher_yolo).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    loader, info = make_yolo_dataloader(
        repair_yaml,
        split="train",
        imgsz=int(cfg.imgsz),
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=0,
        max_images=None,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    log_path = out_dir / "oda_postnms_train_log.csv"
    fields = ["epoch", "step", "loss_total", "loss_task", "loss_oda", "loss_oga"]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    candidate_rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        student.train()
        for batch in loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
                pred = raw_prediction(student, batch["img"])
                with torch.no_grad():
                    teacher_pred = raw_prediction(teacher, batch["img"]) if teacher is not None else None
                loss_oda = matched_candidate_oda_loss(
                    pred,
                    batch,
                    target_ids,
                    teacher_prediction=teacher_pred,
                    iou_threshold=float(cfg.oda_iou_threshold),
                    center_radius=float(cfg.oda_center_radius),
                    topk=int(cfg.oda_topk),
                    cls_weight=float(cfg.oda_cls_weight),
                    box_weight=float(cfg.oda_box_weight),
                    teacher_score_weight=float(cfg.oda_teacher_score_weight),
                    teacher_box_weight=float(cfg.oda_teacher_box_weight),
                    negative_other_cls_weight=float(cfg.negative_other_cls_weight),
                    min_score=float(cfg.oda_min_score),
                    best_score_weight=float(cfg.oda_best_score_weight),
                    best_box_weight=float(cfg.oda_best_box_weight),
                    localized_margin=float(cfg.oda_localized_margin),
                    localized_margin_weight=float(cfg.oda_localized_margin_weight),
                ) * float(cfg.lambda_oda_matched)
                loss_task = supervised_yolo_loss(student, batch) * float(cfg.lambda_task) if cfg.lambda_task > 0 else loss_oda * 0.0
                loss_oga = negative_target_candidate_suppression_loss(
                    pred,
                    batch,
                    target_ids,
                    topk=256,
                    weight=1.0,
                ) * float(cfg.lambda_oga_negative) if cfg.lambda_oga_negative > 0 else loss_oda * 0.0
                loss_total = loss_oda + loss_task + loss_oga
            scaler.scale(loss_total).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss_total": float(loss_total.detach().cpu().item()),
                        "loss_task": float(loss_task.detach().cpu().item()),
                        "loss_oda": float(loss_oda.detach().cpu().item()),
                        "loss_oga": float(loss_oga.detach().cpu().item()),
                    }
                )

        ckpt = out_dir / "02_repair_checkpoints" / f"epoch_{epoch}.pt"
        save_ultralytics_yolo(yolo, ckpt)
        result = run_external_hard_suite_for_yolo(
            str(ckpt),
            data_yaml=cfg.data_yaml,
            target_classes=cfg.target_classes,
            cfg=eval_cfg,
            device=cfg.device,
        )
        eval_dir = out_dir / "03_candidate_external" / f"epoch_{epoch:03d}"
        cand_json, cand_csv = write_external_hard_suite_outputs(result, eval_dir)
        blocked = _blocked_by_worsening(result, before, float(cfg.max_single_attack_worsen))
        summary = result.get("summary") or {}
        row = {
            "epoch": epoch,
            "model": str(ckpt),
            "external_json": str(cand_json),
            "external_rows_csv": str(cand_csv),
            "external_max_asr": float(summary.get("max_asr", 1.0)),
            "external_mean_asr": float(summary.get("mean_asr", 1.0)),
            "score": _external_score(result),
            "blocked_attacks": blocked,
            "accepted": (not blocked) and float(summary.get("max_asr", 1.0)) <= float(cfg.max_allowed_external_asr),
        }
        candidate_rows.append(row)
        write_json(out_dir / "oda_postnms_repair_manifest.json", {"status": "running", "candidate_rows": candidate_rows})

    selection = select_postnms_candidate(
        candidate_rows,
        baseline_score=baseline_score,
        fallback_model=cfg.model,
        min_improvement=float(cfg.min_external_score_improvement),
        require_improvement=bool(cfg.require_improvement_for_final),
    )
    final_row = selection["best"]
    manifest = {
        "status": "passed" if final_row and final_row.get("accepted") else "failed_external_asr_or_worsening",
        "final_model": selection["final_model"],
        "rolled_back": bool(selection["rolled_back"]),
        "input_model": cfg.model,
        "target_class_ids": target_ids,
        "selected_attacks": selected_attacks,
        "before_external_json": str(before_json),
        "before_rows_csv": str(before_csv),
        "before_summary": before.get("summary"),
        "before_score": baseline_score,
        "repair_data_yaml": str(repair_yaml),
        "replay_stats": replay_stats,
        "clean_anchor_stats": clean_stats,
        "n_failure_rows": len(failure_rows),
        "log_csv": str(log_path),
        "candidate_rows": candidate_rows,
        "best": final_row,
        "best_by_score": selection["best_by_score"],
    }
    write_json(out_dir / "oda_postnms_repair_manifest.json", manifest)
    return manifest
