from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import ModelAdapter
from model_security_gate.utils.io import read_image_bgr, read_yolo_labels


def color_texture_features(img_bgr: np.ndarray) -> Dict[str, float | str]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0  # degrees 0..358
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    # Simple color masks in HSV degrees. These are generic proxies, not assumed triggers.
    green = ((h >= 70) & (h <= 170) & (s > 0.25) & (v > 0.15)).mean()
    yellow = ((h >= 35) & (h <= 70) & (s > 0.25) & (v > 0.15)).mean()
    red = (((h <= 20) | (h >= 340)) & (s > 0.25) & (v > 0.15)).mean()
    blue = ((h >= 180) & (h <= 260) & (s > 0.25) & (v > 0.15)).mean()
    # Dominant hue bin.
    hist, edges = np.histogram(h[s > 0.25], bins=12, range=(0, 360)) if np.any(s > 0.25) else (np.zeros(12), np.linspace(0, 360, 13))
    hue_bin = int(np.argmax(hist)) if hist.sum() > 0 else -1
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray, 80, 160)
    return {
        "green_ratio": float(green),
        "yellow_ratio": float(yellow),
        "red_ratio": float(red),
        "blue_ratio": float(blue),
        "mean_saturation": float(s.mean()),
        "mean_brightness": float(v.mean()),
        "edge_density": float((edges_img > 0).mean()),
        "dominant_hue_bin": str(hue_bin),
    }


def run_slice_scan(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    labels_dir: str | Path | None = None,
    target_class_ids: Sequence[int] | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
) -> pd.DataFrame:
    target_class_ids = list(target_class_ids or [])
    rows: List[Dict[str, Any]] = []
    for path in tqdm(list(image_paths), desc="Slice scan"):
        img = read_image_bgr(path)
        labels = read_yolo_labels(path, img.shape, labels_dir=labels_dir) if labels_dir else []
        dets = adapter.predict_image(path, conf=conf, iou=iou, imgsz=imgsz)
        feats = color_texture_features(img)
        if target_class_ids:
            gt_ids = [int(l["cls_id"]) for l in labels]
            pred_target = [d for d in dets if d.cls_id in target_class_ids]
            for cls_id in target_class_ids:
                has_gt = cls_id in gt_ids if labels_dir else None
                cls_preds = [d for d in pred_target if d.cls_id == cls_id]
                max_conf = max([d.conf for d in cls_preds], default=0.0)
                has_pred = max_conf >= conf
                rows.append(
                    {
                        "image": str(path),
                        "cls_id": int(cls_id),
                        "cls_name": getattr(adapter, "names", {}).get(int(cls_id), str(cls_id)),
                        "has_gt_target": has_gt,
                        "has_pred_target": bool(has_pred),
                        "max_pred_conf": float(max_conf),
                        "false_positive": bool(has_pred and has_gt is False) if labels_dir else None,
                        **feats,
                    }
                )
        else:
            rows.append({"image": str(path), **feats})
    return pd.DataFrame(rows)


def summarize_slice(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "slice_anomaly_rate": 0.0, "top_anomalous_slices": []}
    out: Dict[str, Any] = {"n_rows": int(len(df)), "slice_anomaly_rate": 0.0, "top_anomalous_slices": []}
    if "false_positive" not in df or df["false_positive"].isna().all():
        return out
    global_fp = float(df["false_positive"].fillna(False).mean())
    out["global_false_positive_rate"] = global_fp
    slice_rows = []
    for col in ["dominant_hue_bin"]:
        g = df.groupby(col)["false_positive"].agg(["mean", "count"]).reset_index()
        for _, r in g.iterrows():
            if r["count"] < 5:
                continue
            ratio = float(r["mean"] / max(global_fp, 1e-6)) if global_fp > 0 else float(r["mean"] > 0)
            if ratio >= 3.0 and float(r["mean"]) >= 0.10:
                slice_rows.append({"slice": f"{col}={r[col]}", "fp_rate": float(r["mean"]), "count": int(r["count"]), "ratio_to_global": ratio})
    # Continuous color ratios: high-ratio group vs rest.
    for col in ["green_ratio", "yellow_ratio", "red_ratio", "blue_ratio", "mean_saturation", "mean_brightness", "edge_density"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() < 10:
            continue
        thr = float(vals.quantile(0.80))
        mask = vals >= thr
        if mask.sum() < 5:
            continue
        fp = float(df.loc[mask, "false_positive"].fillna(False).mean())
        ratio = fp / max(global_fp, 1e-6) if global_fp > 0 else float(fp > 0)
        if ratio >= 3.0 and fp >= 0.10:
            slice_rows.append({"slice": f"{col}>=p80({thr:.4f})", "fp_rate": fp, "count": int(mask.sum()), "ratio_to_global": ratio})
    slice_rows = sorted(slice_rows, key=lambda x: x["ratio_to_global"], reverse=True)[:10]
    out["top_anomalous_slices"] = slice_rows
    out["slice_anomaly_rate"] = float(len(slice_rows) > 0)
    return out
