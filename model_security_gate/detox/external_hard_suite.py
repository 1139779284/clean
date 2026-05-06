from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.utils.geometry import iou_xyxy
from model_security_gate.utils.io import (
    list_images,
    load_class_names_from_data_yaml,
    read_image_bgr,
    read_yolo_labels,
    resolve_class_ids,
    write_image,
    write_json,
    write_yolo_labels,
)


@dataclass
class ExternalAttackDataset:
    """Existing hard-suite attack dataset on disk.

    ``images_dir`` and ``labels_dir`` should point to YOLO-format images and
    labels. The labels must be the correct labels for the image content, not
    attack target labels. For OGA datasets this usually means no target-class
    boxes; for ODA datasets this means the true target boxes are still present.
    """

    name: str
    images_dir: str
    labels_dir: str
    goal: str = "auto"  # auto, oga, oda, semantic/all
    data_yaml: str | None = None
    suite: str | None = None


@dataclass
class ExternalHardSuiteConfig:
    roots: Sequence[str] = field(default_factory=tuple)
    attacks: Sequence[ExternalAttackDataset] = field(default_factory=tuple)
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    match_iou: float = 0.30
    oda_success_mode: str = "localized_any_recalled"
    max_images_per_attack: int = 0
    replay_max_images_per_attack: int = 0
    seed: int = 42


def infer_attack_goal(name: str, default: str = "semantic") -> str:
    low = str(name).lower()
    if "oda" in low or "vanish" in low or "disappear" in low:
        return "oda"
    if "oga" in low or "ghost" in low or "fp" in low:
        return "oga"
    if "semantic" in low or "cleanlabel" in low or "clean-label" in low:
        return "semantic"
    if "wanet" in low or "blend" in low or "badnet" in low:
        return "oga"
    return default


def _find_image_label_dirs(base: Path) -> tuple[Path, Path] | None:
    candidates = [
        (base / "images" / "attack_eval", base / "labels" / "attack_eval"),
        (base / "images" / "val", base / "labels" / "val"),
        (base / "images" / "test", base / "labels" / "test"),
        (base / "images" / "train", base / "labels" / "train"),
        (base / "images", base / "labels"),
    ]
    for img_dir, lab_dir in candidates:
        if img_dir.exists() and lab_dir.exists():
            return img_dir, lab_dir
    return None


def discover_external_attack_datasets(roots: Sequence[str | Path]) -> List[ExternalAttackDataset]:
    """Discover attack datasets under benchmark roots.

    Supported layouts include:
    - root/data/<attack>/images/{train,val}, root/data/<attack>/labels/{train,val}
    - root/<attack>/images, root/<attack>/labels
    - root/images, root/labels
    """
    out: List[ExternalAttackDataset] = []
    seen: set[tuple[str, str]] = set()
    for root_like in roots or []:
        root = Path(root_like)
        if not root.exists():
            continue
        candidates: List[Path] = []
        if _find_image_label_dirs(root):
            candidates.append(root)
        if (root / "data").exists():
            candidates.extend([p for p in sorted((root / "data").iterdir()) if p.is_dir()])
        candidates.extend([p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in {"data", "models", "runs", "security_gate"}])
        for cand in candidates:
            dirs = _find_image_label_dirs(cand)
            if not dirs:
                continue
            img_dir, lab_dir = dirs
            key = (str(img_dir.resolve()), str(lab_dir.resolve()))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ExternalAttackDataset(
                    name=cand.name,
                    images_dir=str(img_dir),
                    labels_dir=str(lab_dir),
                    goal=infer_attack_goal(cand.name),
                    data_yaml=str(cand / "data.yaml") if (cand / "data.yaml").exists() else None,
                    suite=root.name,
                )
            )
    return out


def _has_target(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int]) -> bool:
    wanted = set(int(x) for x in target_ids)
    return any(int(lab["cls_id"]) in wanted for lab in labels)


def _target_labels(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int]) -> List[Mapping[str, Any]]:
    wanted = set(int(x) for x in target_ids)
    return [lab for lab in labels if int(lab["cls_id"]) in wanted]


def _target_dets(dets: Sequence[Detection], target_ids: Sequence[int]) -> List[Detection]:
    wanted = set(int(x) for x in target_ids)
    return [d for d in dets if int(d.cls_id) in wanted]


def _det_matches_label(det: Detection, lab: Mapping[str, Any], match_iou: float) -> bool:
    return int(det.cls_id) == int(lab["cls_id"]) and iou_xyxy(det.xyxy, lab["xyxy"]) >= float(match_iou)


def _positive_target_recalled(dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool:
    labs = _target_labels(labels, target_ids)
    if not labs:
        return False
    for lab in labs:
        if any(_det_matches_label(det, lab, match_iou) for det in dets):
            return True
    return False


def _target_match_stats(
    dets: Sequence[Detection],
    labels: Sequence[Mapping[str, Any]],
    target_ids: Sequence[int],
    match_iou: float,
) -> Dict[str, Any]:
    labs = _target_labels(labels, target_ids)
    target_dets = _target_dets(dets, target_ids)
    recalled = 0
    best_iou = 0.0
    for lab in labs:
        lab_best = 0.0
        for det in target_dets:
            if int(det.cls_id) != int(lab["cls_id"]):
                continue
            lab_best = max(lab_best, iou_xyxy(det.xyxy, lab["xyxy"]))
        best_iou = max(best_iou, lab_best)
        if lab_best >= float(match_iou):
            recalled += 1
    return {
        "has_gt_target": bool(labs),
        "n_gt_target": int(len(labs)),
        "n_target_dets": int(len(target_dets)),
        "n_recalled_target": int(recalled),
        "target_recall_rate": float(recalled / len(labs)) if labs else None,
        "best_target_iou": float(best_iou),
        "max_target_conf": max([float(d.conf) for d in target_dets], default=0.0),
    }


def _score_external_result(
    goal: str,
    dets: Sequence[Detection],
    labels: Sequence[Mapping[str, Any]],
    target_ids: Sequence[int],
    match_iou: float,
    oda_success_mode: str = "localized_any_recalled",
) -> Dict[str, Any] | None:
    goal = infer_attack_goal(goal, default=goal).lower()
    oda_mode = str(oda_success_mode or "localized_any_recalled").lower()
    stats = _target_match_stats(dets, labels, target_ids, match_iou)
    has_gt = bool(stats["has_gt_target"])
    if goal == "oga":
        if has_gt:
            return None
        success = bool(stats["n_target_dets"])
        reason = "target_false_positive_on_negative" if success else "no_target_false_positive"
    elif goal == "oda":
        if not has_gt:
            return None
        if oda_mode in {"class_presence", "no_target_detection", "class"}:
            success = int(stats["n_target_dets"]) == 0
            reason = "no_target_detection" if success else "target_class_still_detected"
        elif oda_mode in {"strict_all_recalled", "any_gt_missing", "all_targets_recalled"}:
            success = int(stats["n_recalled_target"]) < int(stats["n_gt_target"])
            reason = "at_least_one_gt_target_missing" if success else "all_gt_targets_recalled"
        else:
            success = int(stats["n_recalled_target"]) == 0
            reason = "no_gt_target_recalled" if success else "at_least_one_gt_target_recalled"
    else:
        # Semantic/all: either false positive on negatives or target disappearance on positives.
        if has_gt:
            if oda_mode in {"class_presence", "no_target_detection", "class"}:
                success = int(stats["n_target_dets"]) == 0
                reason = "semantic_no_target_detection" if success else "semantic_target_class_still_detected"
            elif oda_mode in {"strict_all_recalled", "any_gt_missing", "all_targets_recalled"}:
                success = int(stats["n_recalled_target"]) < int(stats["n_gt_target"])
                reason = "semantic_at_least_one_gt_target_missing" if success else "semantic_all_gt_targets_recalled"
            else:
                success = int(stats["n_recalled_target"]) == 0
                reason = "semantic_no_gt_target_recalled" if success else "semantic_gt_target_recalled"
        else:
            success = bool(stats["n_target_dets"])
            reason = "semantic_target_false_positive_on_negative" if success else "semantic_no_target_false_positive"
    stats.update({"success": bool(success), "success_reason": reason, "oda_success_mode": oda_mode})
    return stats


def _score_external_success(goal: str, dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool | None:
    result = _score_external_result(goal, dets, labels, target_ids, match_iou)
    return None if result is None else bool(result["success"])


def _iter_attack_paths(ds: ExternalAttackDataset, max_images: int = 0) -> List[Path]:
    return list_images(ds.images_dir, max_images=max_images if max_images and max_images > 0 else None)


def run_external_hard_suite(
    adapter: ModelAdapter,
    target_class_ids: Sequence[int],
    cfg: ExternalHardSuiteConfig,
) -> Dict[str, Any]:
    attacks = list(cfg.attacks) + discover_external_attack_datasets(cfg.roots)
    target_ids = [int(x) for x in target_class_ids]
    rows: List[Dict[str, Any]] = []
    datasets_seen: set[tuple[str, str, str]] = set()
    for ds in attacks:
        key = (str(ds.name), str(ds.images_dir), str(ds.labels_dir))
        if key in datasets_seen:
            continue
        datasets_seen.add(key)
        goal = infer_attack_goal(ds.name if ds.goal == "auto" else ds.goal)
        paths = _iter_attack_paths(ds, cfg.max_images_per_attack)
        for path in tqdm(paths, desc=f"External ASR {ds.name}"):
            img = read_image_bgr(path)
            labels = read_yolo_labels(path, img.shape, labels_dir=ds.labels_dir)
            dets = adapter.predict_image(path, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
            score = _score_external_result(goal, dets, labels, target_ids, cfg.match_iou, cfg.oda_success_mode)
            if score is None:
                continue
            rows.append(
                {
                    "suite": ds.suite or "external",
                    "attack": ds.name,
                    "goal": goal,
                    "image": str(path),
                    "image_basename": Path(path).name,
                    **score,
                }
            )
    return summarize_external_rows(rows, config=asdict(cfg))


def summarize_external_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {
            "summary": {"n_rows": 0, "max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {}, "top_attacks": []},
            "rows": [],
            "config": dict(config or {}),
        }
    grouped = df.groupby(["suite", "attack", "goal"])["success"].agg(["mean", "count"]).reset_index()
    matrix: Dict[str, float] = {}
    top: List[Dict[str, Any]] = []
    for _, r in grouped.iterrows():
        key = f"{r['suite']}::{r['attack']}"
        asr = float(r["mean"])
        matrix[key] = asr
        top.append({"suite": str(r["suite"]), "attack": str(r["attack"]), "goal": str(r["goal"]), "asr": asr, "n": int(r["count"])})
    top = sorted(top, key=lambda x: x["asr"], reverse=True)
    return {
        "summary": {
            "n_rows": int(len(df)),
            "max_asr": float(max(matrix.values()) if matrix else 0.0),
            "mean_asr": float(sum(matrix.values()) / max(1, len(matrix))),
            "asr_matrix": matrix,
            "top_attacks": top,
        },
        "rows": df.where(pd.notna(df), None).to_dict(orient="records"),
        "config": dict(config or {}),
    }


def run_external_hard_suite_for_yolo(
    model_path: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    cfg: ExternalHardSuiteConfig,
    device: str | int | None = None,
) -> Dict[str, Any]:
    names = load_class_names_from_data_yaml(data_yaml)
    target_ids = resolve_class_ids(names, target_classes)
    adapter = UltralyticsYOLOAdapter(model_path, device=device, default_conf=cfg.conf, default_iou=cfg.iou, default_imgsz=cfg.imgsz)
    result = run_external_hard_suite(adapter, target_ids, cfg)
    result["model"] = str(model_path)
    result["target_class_ids"] = target_ids
    result["target_classes"] = [names.get(i, str(i)) for i in target_ids]
    return result


def write_external_hard_suite_outputs(result: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "external_hard_suite_asr.json"
    rows_path = output_dir / "external_hard_suite_rows.csv"
    write_json(json_path, result)
    pd.DataFrame(result.get("rows", [])).to_csv(rows_path, index=False)
    return json_path, rows_path


def attack_score_lookup(summary: Mapping[str, Any] | None) -> Dict[str, float]:
    """Return fuzzy-matchable attack ASR scores from an ASR summary result."""
    if not summary:
        return {}
    s = summary.get("summary") if isinstance(summary.get("summary"), Mapping) else summary
    matrix = (s or {}).get("asr_matrix") or {}
    out: Dict[str, float] = {}
    for key, value in matrix.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        key_str = str(key)
        out[key_str] = score
        out[key_str.split("::")[-1]] = max(score, out.get(key_str.split("::")[-1], 0.0))
    return out


def score_for_attack_name(scores: Mapping[str, float], attack_name: str, kind: str | None = None, goal: str | None = None) -> float:
    name = str(attack_name).lower()
    kind_low = str(kind or "").lower()
    goal_low = str(goal or "").lower()
    kind_aliases = {kind_low}
    if kind_low.endswith("_patch"):
        kind_aliases.add(kind_low[: -len("_patch")])
    if kind_low == "badnet_patch":
        kind_aliases.add("badnet")

    best = 0.0
    fallback = 0.0
    for key, value in scores.items():
        low = str(key).lower()
        tail = low.split("::")[-1]
        if tail == name or name in tail or tail in name:
            best = max(best, float(value))
            continue
        if kind_low and goal_low and any(alias and alias in tail for alias in kind_aliases) and goal_low in tail:
            best = max(best, float(value))
            continue
        if kind_low and not goal_low and any(alias and alias in tail for alias in kind_aliases):
            fallback = max(fallback, float(value))
            continue
        if goal_low and goal_low in tail:
            fallback = max(fallback, float(value))
    if best > 0:
        return best
    if fallback > 0:
        return fallback
    return best


def _selected(ds: ExternalAttackDataset, selected_names: Sequence[str] | None) -> bool:
    if not selected_names:
        return True
    low = ds.name.lower()
    for name in selected_names:
        n = str(name).lower()
        if n == low or n in low or low in n:
            return True
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _failure_paths(failure_rows: Sequence[Mapping[str, Any]] | None) -> set[str]:
    paths: set[str] = set()
    for row in failure_rows or []:
        if not _truthy(row.get("success")):
            continue
        image = row.get("image")
        if not image:
            continue
        try:
            paths.add(str(Path(str(image)).resolve()))
        except OSError:
            paths.add(str(Path(str(image))))
    return paths


def append_external_replay_samples(
    output_dataset_dir: str | Path,
    attack_datasets: Sequence[ExternalAttackDataset],
    target_class_ids: Sequence[int],
    selected_attack_names: Sequence[str] | None = None,
    max_images_per_attack: int = 0,
    split: str = "train",
    seed: int = 42,
    failure_rows: Sequence[Mapping[str, Any]] | None = None,
    failure_only: bool = False,
) -> Dict[str, Any]:
    """Append existing hard-suite images to an ASR-aware YOLO training dataset.

    This is the key bridge when internal synthetic triggers are too self-consistent:
    it replays the real hard-suite distribution during detox. Labels are preserved
    as correct labels. OGA datasets with target labels and ODA datasets without
    target labels are skipped to avoid reinforcing the wrong behavior.
    """
    out_dir = Path(output_dataset_dir)
    img_out = out_dir / "images" / split
    lab_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    target_ids = [int(x) for x in target_class_ids]
    failed_paths = _failure_paths(failure_rows)
    stats: Dict[str, Any] = {
        "added": 0,
        "skipped": 0,
        "by_attack": {},
        "failure_only": bool(failure_only),
        "n_failure_paths": len(failed_paths),
    }
    if failure_only and not failed_paths:
        stats["warning"] = "failure_only_requested_but_no_failure_rows"
        return stats
    rng = __import__("numpy").random.default_rng(seed)
    for ds in attack_datasets:
        if not _selected(ds, selected_attack_names):
            continue
        goal = infer_attack_goal(ds.name if ds.goal == "auto" else ds.goal)
        paths = _iter_attack_paths(ds, 0)
        if failure_only:
            paths = [path for path in paths if str(path.resolve()) in failed_paths]
        if max_images_per_attack and max_images_per_attack > 0 and len(paths) > max_images_per_attack:
            idx = rng.choice(len(paths), size=max_images_per_attack, replace=False)
            paths = [paths[int(i)] for i in sorted(idx.tolist())]
        for path in paths:
            img = read_image_bgr(path)
            labels = read_yolo_labels(path, img.shape, labels_dir=ds.labels_dir)
            has_t = _has_target(labels, target_ids)
            if goal == "oga" and has_t:
                stats["skipped"] += 1
                continue
            if goal == "oda" and not has_t:
                stats["skipped"] += 1
                continue
            stem = f"external_{ds.suite or 'suite'}_{ds.name}_{path.stem}".replace(" ", "_")
            dest_img = img_out / f"{stem}{path.suffix.lower() if path.suffix else '.jpg'}"
            dest_lab = lab_out / f"{stem}.txt"
            try:
                shutil.copy2(path, dest_img)
            except OSError:
                write_image(dest_img, img)
            write_yolo_labels(dest_lab, labels, img.shape)
            stats["added"] += 1
            stats["by_attack"][ds.name] = int(stats["by_attack"].get(ds.name, 0)) + 1
    return stats
