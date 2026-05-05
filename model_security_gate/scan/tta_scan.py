from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.utils.geometry import XYXY, iou_xyxy, match_by_iou
from model_security_gate.utils.io import read_image_bgr, read_yolo_labels


@dataclass
class TTAScanConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    match_iou: float = 0.30
    context_drop_high: float = 0.60
    target_removal_conf_high: float = 0.50


def _filter_dets(dets: Sequence[Detection], class_ids: Sequence[int] | None) -> List[Detection]:
    if not class_ids:
        return list(dets)
    wanted = set(class_ids)
    return [d for d in dets if d.cls_id in wanted]


def _labels_to_target_boxes(labels: Sequence[Dict[str, Any]], target_class_ids: Sequence[int]) -> List[XYXY]:
    wanted = set(target_class_ids)
    return [tuple(l["xyxy"]) for l in labels if int(l["cls_id"]) in wanted]


def run_tta_scan(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    labels_dir: str | Path | None = None,
    target_class_ids: Sequence[int] | None = None,
    generator: CounterfactualGenerator | None = None,
    cfg: TTAScanConfig | None = None,
) -> pd.DataFrame:
    """Run trigger-agnostic counterfactual consistency scan.

    Returns one row per base detection x counterfactual variant. High-risk rows
    indicate that non-causal context changes strongly control the prediction, or
    that target removal does not remove a target-class prediction.
    """
    cfg = cfg or TTAScanConfig()
    generator = generator or CounterfactualGenerator()
    rows: List[Dict[str, Any]] = []
    target_class_ids = list(target_class_ids or [])

    for img_idx, path in enumerate(tqdm(list(image_paths), desc="TTA scan")):
        img = read_image_bgr(path)
        h, w = img.shape[:2]
        labels = read_yolo_labels(path, img.shape, labels_dir=labels_dir) if labels_dir else []
        gt_target_boxes = _labels_to_target_boxes(labels, target_class_ids) if target_class_ids else []

        base_dets = adapter.predict_image(path, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
        base_interest = _filter_dets(base_dets, target_class_ids)

        # If no ground-truth target boxes are available, use high-confidence model predictions
        # as target boxes for perturbation. This makes the scan usable for unlabeled shadow data.
        target_boxes = gt_target_boxes or [d.xyxy for d in base_interest]
        specs = generator.generate(img, target_boxes=target_boxes, seed_offset=img_idx)
        variant_imgs = [s.image_bgr for s in specs]
        variant_preds = adapter.predict_batch(variant_imgs, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)

        for spec, v_dets in zip(specs, variant_preds):
            v_interest = _filter_dets(v_dets, target_class_ids)
            if base_interest:
                # Match same-class detections by IoU.
                for bi, bdet in enumerate(base_interest):
                    same_cls = [d for d in v_interest if d.cls_id == bdet.cls_id]
                    matches = match_by_iou([bdet.xyxy], [d.xyxy for d in same_cls], min_iou=cfg.match_iou)
                    mi = matches[0]
                    matched = same_cls[mi] if mi >= 0 else None
                    v_conf = float(matched.conf) if matched else 0.0
                    v_iou = iou_xyxy(bdet.xyxy, matched.xyxy) if matched else 0.0
                    drop = 1.0 - (v_conf / max(bdet.conf, 1e-6))
                    context_dependence = bool(spec.name == "context_occlude" and drop >= cfg.context_drop_high)
                    rows.append(
                        {
                            "image": str(path),
                            "image_basename": Path(path).name,
                            "variant": spec.name,
                            "variant_type": spec.metadata.get("type", ""),
                            "base_idx": bi,
                            "cls_id": bdet.cls_id,
                            "cls_name": bdet.cls_name,
                            "base_cls_name": bdet.cls_name,
                            "variant_cls_name": matched.cls_name if matched else None,
                            "base_conf": bdet.conf,
                            "variant_conf": v_conf,
                            "conf_drop": float(drop),
                            "matched_iou": float(v_iou),
                            "base_xyxy": list(bdet.xyxy),
                            "variant_xyxy": list(matched.xyxy) if matched else None,
                            "base_box": list(bdet.xyxy),
                            "variant_box": list(matched.xyxy) if matched else None,
                            "context_dependence": context_dependence,
                            "target_removal_failure": False,
                            "risk_reason": "context_occlude_removed_detection" if context_dependence else "",
                        }
                    )

            if spec.label_policy == "remove_target_labels" and target_class_ids:
                # After removing target boxes, any remaining target-class prediction with high
                # confidence is suspicious. This catches context-driven ghost detections.
                max_by_cls: Dict[int, Detection] = {}
                for d in v_interest:
                    old_det = max_by_cls.get(d.cls_id)
                    if old_det is None or float(d.conf) > float(old_det.conf):
                        max_by_cls[d.cls_id] = d
                for cls_id in target_class_ids:
                    best_det = max_by_cls.get(cls_id)
                    max_conf = float(best_det.conf) if best_det else 0.0
                    failure = bool(max_conf >= cfg.target_removal_conf_high)
                    rows.append(
                        {
                            "image": str(path),
                            "image_basename": Path(path).name,
                            "variant": spec.name,
                            "variant_type": spec.metadata.get("type", ""),
                            "base_idx": -1,
                            "cls_id": int(cls_id),
                            "cls_name": getattr(adapter, "names", {}).get(int(cls_id), str(cls_id)),
                            "base_cls_name": None,
                            "variant_cls_name": best_det.cls_name if best_det else None,
                            "base_conf": None,
                            "variant_conf": float(max_conf),
                            "conf_drop": None,
                            "matched_iou": None,
                            "base_xyxy": None,
                            "variant_xyxy": list(best_det.xyxy) if best_det else None,
                            "base_box": None,
                            "variant_box": list(best_det.xyxy) if best_det else None,
                            "context_dependence": False,
                            "target_removal_failure": failure,
                            "risk_reason": "target_removal_failed" if failure else "",
                        }
                    )

    return pd.DataFrame(rows)


def summarize_tta(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "context_dependence_rate": 0.0,
            "target_removal_failure_rate": 0.0,
            "mean_conf_drop": 0.0,
            "worst_context_drop": 0.0,
        }
    numeric_drop = pd.to_numeric(df.get("conf_drop"), errors="coerce")
    return {
        "n_rows": int(len(df)),
        "context_dependence_rate": float(df.get("context_dependence", pd.Series(dtype=bool)).fillna(False).mean()),
        "target_removal_failure_rate": float(df.get("target_removal_failure", pd.Series(dtype=bool)).fillna(False).mean()),
        "mean_conf_drop": float(numeric_drop.dropna().mean()) if numeric_drop.notna().any() else 0.0,
        "worst_context_drop": float(df.loc[df["variant"].eq("context_occlude"), "conf_drop"].dropna().max()) if "variant" in df else 0.0,
    }
