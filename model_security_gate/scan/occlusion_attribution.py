from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.utils.geometry import XYXY, clip_xyxy, iou_xyxy, union_mask_from_boxes
from model_security_gate.utils.io import read_image_bgr, write_image


@dataclass
class OcclusionConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    grid: int = 7
    max_detections_per_image: int = 3
    match_iou: float = 0.25
    attribution_overlap_low: float = 0.30
    save_heatmaps: bool = True


def _fill_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = rect
    med = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)
    out[y1:y2, x1:x2] = med
    return out


def _matched_conf(dets: Sequence[Detection], base: Detection, min_iou: float) -> float:
    best = 0.0
    for d in dets:
        if d.cls_id != base.cls_id:
            continue
        if iou_xyxy(d.xyxy, base.xyxy) >= min_iou:
            best = max(best, float(d.conf))
    return best


def _heatmap_box_mass(heatmap: np.ndarray, box: XYXY) -> float:
    h, w = heatmap.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clip_xyxy(box, w, h)]
    total = float(np.maximum(heatmap, 0).sum())
    if total <= 1e-12:
        return 0.0
    inside = float(np.maximum(heatmap[y1 : y2 + 1, x1 : x2 + 1], 0).sum())
    return inside / total


def _save_overlay(path: Path, img: np.ndarray, heatmap: np.ndarray, box: XYXY) -> None:
    h = np.maximum(heatmap, 0)
    if h.max() > 0:
        h = h / h.max()
    h_uint = (h * 255).astype(np.uint8)
    color = cv2.applyColorMap(h_uint, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.62, color, 0.38, 0)
    x1, y1, x2, y2 = [int(round(v)) for v in clip_xyxy(box, img.shape[1], img.shape[0])]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    write_image(path, overlay)


def compute_occlusion_heatmap(
    adapter: ModelAdapter,
    image_bgr: np.ndarray,
    base_det: Detection,
    cfg: OcclusionConfig,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    g = max(2, int(cfg.grid))
    variants: List[np.ndarray] = []
    rects: List[Tuple[int, int, int, int]] = []
    for gy in range(g):
        for gx in range(g):
            x1 = int(round(gx * w / g))
            x2 = int(round((gx + 1) * w / g))
            y1 = int(round(gy * h / g))
            y2 = int(round((gy + 1) * h / g))
            rect = (x1, y1, x2, y2)
            variants.append(_fill_rect(image_bgr, rect))
            rects.append(rect)
    preds = adapter.predict_batch(variants, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
    base_conf = max(float(base_det.conf), 1e-6)
    grid_scores = np.zeros((g, g), dtype=np.float32)
    for idx, dets in enumerate(preds):
        var_conf = _matched_conf(dets, base_det, cfg.match_iou)
        drop = max(0.0, base_conf - var_conf) / base_conf
        gy, gx = divmod(idx, g)
        grid_scores[gy, gx] = drop
    return cv2.resize(grid_scores, (w, h), interpolation=cv2.INTER_CUBIC)


def run_occlusion_attribution_scan(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    target_class_ids: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    cfg: OcclusionConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OcclusionConfig()
    target_class_ids = list(target_class_ids or [])
    out_dir = Path(output_dir) if output_dir else None
    if out_dir and cfg.save_heatmaps:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for path in tqdm(list(image_paths), desc="Occlusion attribution"):
        img = read_image_bgr(path)
        dets = adapter.predict_image(path, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
        if target_class_ids:
            dets = [d for d in dets if d.cls_id in target_class_ids]
        dets = sorted(dets, key=lambda d: d.conf, reverse=True)[: cfg.max_detections_per_image]
        for idx, det in enumerate(dets):
            heatmap = compute_occlusion_heatmap(adapter, img, det, cfg)
            overlap = _heatmap_box_mass(heatmap, det.xyxy)
            suspicious = bool(overlap < cfg.attribution_overlap_low)
            heatmap_path = None
            if out_dir and cfg.save_heatmaps:
                heatmap_path = out_dir / f"{Path(path).stem}_det{idx}_{det.cls_name}_occ.jpg"
                _save_overlay(heatmap_path, img, heatmap, det.xyxy)
            rows.append(
                {
                    "image": str(path),
                    "det_idx": idx,
                    "cls_id": det.cls_id,
                    "cls_name": det.cls_name,
                    "conf": det.conf,
                    "xyxy": list(det.xyxy),
                    "attribution_mass_in_box": float(overlap),
                    "wrong_region_attention": suspicious,
                    "heatmap_path": str(heatmap_path) if heatmap_path else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_occlusion(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "wrong_region_attention_rate": 0.0, "mean_mass_in_box": 0.0}
    return {
        "n_rows": int(len(df)),
        "wrong_region_attention_rate": float(df["wrong_region_attention"].fillna(False).mean()),
        "mean_mass_in_box": float(pd.to_numeric(df["attribution_mass_in_box"], errors="coerce").dropna().mean()),
    }
