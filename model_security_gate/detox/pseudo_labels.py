from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection
from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.utils.geometry import iou_xyxy
from model_security_gate.utils.io import list_images, read_image_bgr, write_image, write_yolo_labels, write_yaml, write_json


@dataclass
class PseudoLabelConfig:
    """Configuration for label-free / pseudo-label detox dataset construction.

    The strict fields below are intentionally separate from the legacy ``conf``
    field so teacher and suspicious model predictions can be filtered
    differently. In agreement mode the teacher is the source of label geometry,
    but a suspicious-model box must agree with it.
    """

    # Legacy/default confidence. Kept for backwards compatibility.
    conf: float = 0.45
    iou: float = 0.7
    imgsz: int = 640
    val_fraction: float = 0.15
    seed: int = 42
    image_ext: str = ".jpg"
    include_original: bool = True
    variants: Sequence[str] | None = None
    target_class_ids: Sequence[int] | None = None
    # teacher: use only teacher predictions; suspicious: use suspicious predictions;
    # agreement: keep boxes only when teacher and suspicious agree by IoU/class/conf-gap.
    source: str = "agreement"
    agreement_iou: float = 0.50
    min_box_area_frac: float = 0.0005
    max_boxes_per_image: int = 100
    keep_non_target_labels: bool = True

    # Strict quality controls requested for unknown-trigger/label-free detox.
    min_teacher_conf: float = 0.45
    min_suspicious_conf: float = 0.25
    max_conf_gap: float = 0.35
    require_class_agreement: bool = True
    reject_if_teacher_empty: bool = True
    save_rejected_samples: bool = True


def _split_paths(paths: Sequence[Path], val_fraction: float, seed: int) -> tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = int(round(len(paths) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, p in enumerate(paths):
        (val if i in val_idx else train).append(p)
    return train, val


def _mean_conf(dets: Sequence[Detection]) -> float | None:
    if not dets:
        return None
    return float(np.mean([float(d.conf) for d in dets]))


def _detections_to_labels(
    dets: Sequence[Detection],
    image_shape: tuple[int, int, int] | tuple[int, int],
    class_ids: Sequence[int] | None = None,
    min_area_frac: float = 0.0,
    max_boxes: int = 100,
) -> List[Dict[str, Any]]:
    h, w = image_shape[:2]
    wanted = set(int(x) for x in class_ids or [])
    out: List[Dict[str, Any]] = []
    for d in sorted(dets, key=lambda x: float(x.conf), reverse=True):
        if wanted and int(d.cls_id) not in wanted:
            continue
        x1, y1, x2, y2 = [float(v) for v in d.xyxy]
        area_frac = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(w * h))
        if area_frac < min_area_frac:
            continue
        out.append(
            {
                "cls_id": int(d.cls_id),
                "xyxy": (x1, y1, x2, y2),
                "conf": float(d.conf),
                "source": "pseudo",
            }
        )
        if len(out) >= max_boxes:
            break
    return out


def _best_match(
    td: Detection,
    suspicious_dets: Sequence[Detection],
    require_class_agreement: bool,
) -> tuple[Detection | None, float]:
    best: Detection | None = None
    best_iou = -1.0
    for sd in suspicious_dets:
        if require_class_agreement and int(sd.cls_id) != int(td.cls_id):
            continue
        overlap = iou_xyxy(td.xyxy, sd.xyxy)
        if overlap > best_iou:
            best = sd
            best_iou = overlap
    return best, max(0.0, float(best_iou))


def _agreement_filter(
    teacher_dets: Sequence[Detection],
    suspicious_dets: Sequence[Detection],
    cfg: PseudoLabelConfig,
) -> tuple[List[Detection], Dict[str, Any]]:
    """Keep teacher boxes that pass class/IoU/confidence-gap agreement."""
    kept: List[Detection] = []
    rejected_iou = 0
    rejected_class = 0
    rejected_conf_gap = 0
    matched_any = 0
    for td in teacher_dets:
        sd, overlap = _best_match(td, suspicious_dets, require_class_agreement=cfg.require_class_agreement)
        if sd is None:
            rejected_class += 1
            continue
        matched_any += 1
        if overlap < float(cfg.agreement_iou):
            rejected_iou += 1
            continue
        if abs(float(td.conf) - float(sd.conf)) > float(cfg.max_conf_gap):
            rejected_conf_gap += 1
            continue
        kept.append(td)
    details = {
        "agreement_matches": len(kept),
        "teacher_boxes_considered": len(teacher_dets),
        "suspicious_boxes_considered": len(suspicious_dets),
        "matched_any": matched_any,
        "rejected_iou": rejected_iou,
        "rejected_class_or_no_match": rejected_class,
        "rejected_conf_gap": rejected_conf_gap,
        "agreement_rate": float(len(kept) / max(1, len(teacher_dets))),
    }
    return kept, details


def _remove_target_labels(labels: List[Dict[str, Any]], target_class_ids: Sequence[int]) -> List[Dict[str, Any]]:
    wanted = set(int(x) for x in target_class_ids)
    if not wanted:
        # Unknown target class: object-removal variants remove all pseudo labels.
        return []
    return [lab for lab in labels if int(lab["cls_id"]) not in wanted]


def _target_boxes_from_labels(labels: Sequence[Dict[str, Any]], target_class_ids: Sequence[int] | None) -> List[tuple[float, float, float, float]]:
    wanted = set(int(x) for x in target_class_ids or [])
    if not wanted:
        return [tuple(lab["xyxy"]) for lab in labels]
    return [tuple(lab["xyxy"]) for lab in labels if int(lab["cls_id"]) in wanted]


def make_pseudo_labels_with_quality(
    suspicious_adapter: UltralyticsYOLOAdapter,
    image: str | Path,
    image_shape: tuple[int, int, int] | tuple[int, int],
    teacher_adapter: UltralyticsYOLOAdapter | None = None,
    cfg: PseudoLabelConfig | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = cfg or PseudoLabelConfig()
    source = cfg.source.lower().strip()
    if source not in {"teacher", "agreement", "suspicious"}:
        raise ValueError("PseudoLabelConfig.source must be teacher, agreement, or suspicious")

    suspicious_dets = suspicious_adapter.predict_image(
        image,
        conf=cfg.min_suspicious_conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
    )
    teacher_dets: List[Detection] = []
    if teacher_adapter is not None:
        teacher_dets = teacher_adapter.predict_image(
            image,
            conf=cfg.min_teacher_conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
        )

    quality: Dict[str, Any] = {
        "image": str(image),
        "image_basename": Path(image).name,
        "source": source,
        "has_teacher": teacher_adapter is not None,
        "n_teacher_dets": len(teacher_dets),
        "n_suspicious_dets": len(suspicious_dets),
        "mean_teacher_conf": _mean_conf(teacher_dets),
        "mean_suspicious_conf": _mean_conf(suspicious_dets),
        "accepted": True,
        "rejected_reason": "",
        "agreement_rate": None,
        "agreement_matches": None,
        "n_pseudo_boxes": 0,
        "empty_label": False,
    }

    if teacher_adapter is not None and source in {"teacher", "agreement"} and cfg.reject_if_teacher_empty and not teacher_dets:
        quality.update({"accepted": False, "rejected_reason": "teacher_empty", "empty_label": True})
        return [], quality

    if source == "teacher" and teacher_adapter is not None:
        dets = teacher_dets
        quality.update({"agreement_rate": 1.0 if teacher_dets else 0.0, "agreement_matches": len(teacher_dets)})
    elif source == "agreement" and teacher_adapter is not None:
        dets, details = _agreement_filter(teacher_dets, suspicious_dets, cfg)
        quality.update(details)
        if teacher_dets and not dets:
            quality.update({"accepted": False, "rejected_reason": "no_agreed_boxes"})
            return [], quality
    else:
        dets = suspicious_dets
        quality.update({"agreement_rate": 1.0 if suspicious_dets else 0.0, "agreement_matches": len(suspicious_dets)})

    labels = _detections_to_labels(
        dets,
        image_shape=image_shape,
        class_ids=None,
        min_area_frac=cfg.min_box_area_frac,
        max_boxes=cfg.max_boxes_per_image,
    )
    quality["n_pseudo_boxes"] = len(labels)
    quality["empty_label"] = len(labels) == 0
    if dets and not labels:
        quality.update({"accepted": False, "rejected_reason": "boxes_too_small_or_filtered"})
    return labels, quality


def make_pseudo_labels_for_image(
    suspicious_adapter: UltralyticsYOLOAdapter,
    image: str | Path,
    image_shape: tuple[int, int, int] | tuple[int, int],
    teacher_adapter: UltralyticsYOLOAdapter | None = None,
    cfg: PseudoLabelConfig | None = None,
) -> List[Dict[str, Any]]:
    labels, _quality = make_pseudo_labels_with_quality(suspicious_adapter, image, image_shape, teacher_adapter, cfg)
    return labels


def summarize_pseudo_label_quality(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_images": 0,
            "n_pseudo_boxes": 0,
            "n_rejected": 0,
            "mean_teacher_conf": 0.0,
            "agreement_rate": 0.0,
            "empty_label_rate": 0.0,
        }
    df = pd.DataFrame(rows)
    accepted = df.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool)
    n_images = int(len(df))
    n_accepted = int(accepted.sum())
    n_rejected = int((~accepted).sum())
    n_boxes = int(pd.to_numeric(df.get("n_pseudo_boxes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    def mean_col(name: str) -> float:
        if name not in df:
            return 0.0
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        return float(s.mean()) if not s.empty else 0.0

    reason_counts = Counter(str(x) for x in df.loc[~accepted, "rejected_reason"].fillna("unknown")) if "rejected_reason" in df else Counter()
    empty_label = df.get("empty_label", pd.Series([False] * n_images)).fillna(False).astype(bool)
    return {
        "n_images": n_images,
        "n_accepted": n_accepted,
        "n_pseudo_boxes": n_boxes,
        "n_rejected": n_rejected,
        "mean_teacher_conf": mean_col("mean_teacher_conf"),
        "mean_suspicious_conf": mean_col("mean_suspicious_conf"),
        "agreement_rate": mean_col("agreement_rate"),
        "empty_label_rate": float(empty_label.mean()) if n_images else 0.0,
        "rejected_rate": float(n_rejected / max(1, n_images)),
        "boxes_per_accepted_image": float(n_boxes / max(1, n_accepted)),
        "rejection_reasons": dict(reason_counts),
    }


def _write_quality_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_pseudo_counterfactual_yolo_dataset(
    suspicious_model: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    class_names: Dict[int, str] | Sequence[str],
    teacher_model: str | Path | None = None,
    cfg: PseudoLabelConfig | None = None,
    device: str | int | None = None,
) -> Path:
    """Build a YOLO detox dataset without human labels.

    Label policy:
    - Original and non-object-removal variants keep conservative pseudo labels.
    - target_occlude / target_inpaint remove target_class_ids. If target_class_ids
      is empty, remove all pseudo labels for that image. This makes unknown-target
      mode possible but noisier.

    Recommended use:
    - If you have a clean teacher, use source='teacher' or 'agreement'.
    - If you have no trusted teacher, use source='suspicious' only as a weak
      fallback and rely more on NAD/IBAU feature detox than supervised CF loss.
    """
    cfg = cfg or PseudoLabelConfig()
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    pseudo_raw_dir = output_dir / "pseudo_raw_labels"
    rejected_dir = output_dir / "rejected_samples"
    for split in ["train", "val"]:
        (images_out / split).mkdir(parents=True, exist_ok=True)
        (labels_out / split).mkdir(parents=True, exist_ok=True)
    pseudo_raw_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_rejected_samples:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    suspicious_adapter = UltralyticsYOLOAdapter(
        suspicious_model,
        device=device,
        default_conf=cfg.min_suspicious_conf,
        default_iou=cfg.iou,
        default_imgsz=cfg.imgsz,
    )
    teacher_adapter = (
        UltralyticsYOLOAdapter(
            teacher_model,
            device=device,
            default_conf=cfg.min_teacher_conf,
            default_iou=cfg.iou,
            default_imgsz=cfg.imgsz,
        )
        if teacher_model
        else None
    )
    paths = list_images(images_dir)
    train_paths, val_paths = _split_paths(paths, cfg.val_fraction, cfg.seed)
    variants = cfg.variants or [
        "grayscale",
        "low_saturation",
        "hue_rotate",
        "brightness_contrast",
        "jpeg",
        "blur",
        "random_patch",
        "context_occlude",
        "target_occlude",
        "target_inpaint",
    ]
    generator = CounterfactualGenerator(variants=variants, seed=cfg.seed)
    stats: Dict[str, Any] = {
        "train": 0,
        "val": 0,
        "pseudo_boxes": 0,
        "images_with_no_pseudo_labels": 0,
        "rejected_images": 0,
    }
    quality_rows: list[dict] = []

    for split, split_paths in [("train", train_paths), ("val", val_paths)]:
        for img_idx, img_path in enumerate(tqdm(split_paths, desc=f"Build pseudo-CF {split}")):
            img = read_image_bgr(img_path)
            labels, quality = make_pseudo_labels_with_quality(suspicious_adapter, img_path, img.shape, teacher_adapter, cfg)
            quality["split"] = split
            quality_rows.append(quality)
            if not quality.get("accepted", True):
                stats["rejected_images"] += 1
                if cfg.save_rejected_samples:
                    target = rejected_dir / split / img_path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(img_path, target)
                    except OSError:
                        write_image(target, img)
                continue

            stats["pseudo_boxes"] += len(labels)
            if not labels:
                stats["images_with_no_pseudo_labels"] += 1
            # Store raw pseudo labels for inspection.
            write_yolo_labels(pseudo_raw_dir / f"{img_path.stem}.txt", labels, img.shape)

            target_boxes = _target_boxes_from_labels(labels, cfg.target_class_ids)
            base_stem = img_path.stem
            if cfg.include_original:
                out_img = images_out / split / f"{base_stem}_orig{cfg.image_ext}"
                out_lab = labels_out / split / f"{base_stem}_orig.txt"
                write_image(out_img, img)
                write_yolo_labels(out_lab, labels, img.shape)
                stats[split] += 1
            specs = generator.generate(img, target_boxes=target_boxes, seed_offset=img_idx)
            for spec in specs:
                out_img = images_out / split / f"{base_stem}_{spec.name}{cfg.image_ext}"
                out_lab = labels_out / split / f"{base_stem}_{spec.name}.txt"
                write_image(out_img, spec.image_bgr)
                if spec.label_policy == "remove_target_labels":
                    v_labels = _remove_target_labels(labels, cfg.target_class_ids or [])
                else:
                    v_labels = labels
                write_yolo_labels(out_lab, v_labels, spec.image_bgr.shape)
                stats[split] += 1

    quality_csv = output_dir / "pseudo_label_quality.csv"
    _write_quality_csv(quality_csv, quality_rows)
    quality_summary = summarize_pseudo_label_quality(quality_rows)

    if isinstance(class_names, dict):
        names = {int(k): str(v) for k, v in class_names.items()}
        names_list = [names[i] for i in sorted(names)]
    else:
        names_list = [str(x) for x in class_names]
    data_yaml = output_dir / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names_list,
            "detox_stats": stats,
            "label_mode": "pseudo",
            "pseudo_label_config": asdict(cfg),
            "pseudo_quality_summary": quality_summary,
            "pseudo_quality_csv": str(quality_csv),
        },
    )
    write_json(
        output_dir / "pseudo_label_manifest.json",
        {
            "config": asdict(cfg),
            "stats": stats,
            "quality_summary": quality_summary,
            "quality_csv": str(quality_csv),
            "teacher_model": str(teacher_model) if teacher_model else None,
        },
    )
    return data_yaml
