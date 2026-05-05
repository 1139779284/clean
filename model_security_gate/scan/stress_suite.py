from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.cf.transforms import hue_rotate, jpeg_compress, low_saturation, random_patch_occlude
from model_security_gate.utils.io import read_image_bgr, read_yolo_labels


def _sinusoidal_overlay(img: np.ndarray, amplitude: float = 10.0, freq: int = 8, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    coord = np.cos(angle) * xx + np.sin(angle) * yy
    wave = np.sin(2 * np.pi * freq * coord / max(h, w))
    noise = amplitude * wave[..., None]
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _lowfreq_noise(img: np.ndarray, amplitude: float = 12.0, grid: int = 8, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    small = rng.normal(0, amplitude, size=(grid, grid, 3)).astype(np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _max_target_det(dets: Sequence[Detection], target_class_ids: Sequence[int]) -> Dict[int, Detection | None]:
    out: Dict[int, Detection | None] = {int(c): None for c in target_class_ids}
    for d in dets:
        if d.cls_id in out:
            old = out[d.cls_id]
            if old is None or float(d.conf) > float(old.conf):
                out[d.cls_id] = d
    return out


def _det_conf(det: Detection | None) -> float:
    return float(det.conf) if det else 0.0


def _det_box(det: Detection | None) -> list[float] | None:
    return list(det.xyxy) if det else None


def _det_cls_name(det: Detection | None, names: Dict[int, str], cls_id: int) -> str | None:
    if det:
        return det.cls_name
    return names.get(int(cls_id), str(cls_id))


def run_stress_suite(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    labels_dir: str | Path | None = None,
    target_class_ids: Sequence[int] | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    n_random_patches: int = 5,
    inflation_threshold: float = 0.35,
) -> pd.DataFrame:
    target_class_ids = list(target_class_ids or [])
    if not target_class_ids:
        raise ValueError("run_stress_suite requires target_class_ids")
    rows: List[Dict[str, Any]] = []
    names = getattr(adapter, "names", {}) or {}
    for idx, path in enumerate(tqdm(list(image_paths), desc="Stress suite")):
        img = read_image_bgr(path)
        labels = read_yolo_labels(path, img.shape, labels_dir=labels_dir) if labels_dir else []
        gt_ids = [int(l["cls_id"]) for l in labels]
        base = adapter.predict_image(path, conf=conf, iou=iou, imgsz=imgsz)
        base_dets = _max_target_det(base, target_class_ids)
        variants: List[tuple[str, np.ndarray]] = [
            ("hue_rotate", hue_rotate(img, 90)),
            ("low_saturation", low_saturation(img, 0.05)),
            ("jpeg_lowq", jpeg_compress(img, 18)),
            ("sinusoidal_low_amp", _sinusoidal_overlay(img, amplitude=8.0, freq=6, seed=idx)),
            ("lowfreq_noise", _lowfreq_noise(img, amplitude=10.0, grid=7, seed=idx)),
        ]
        for k in range(n_random_patches):
            variants.append((f"random_patch_{k}", random_patch_occlude(img, n=1, patch_frac=0.10 + 0.02 * (k % 3), seed=idx * 100 + k)))
        preds = adapter.predict_batch([v[1] for v in variants], conf=conf, iou=iou, imgsz=imgsz)
        for (name, _), dets in zip(variants, preds):
            variant_dets = _max_target_det(dets, target_class_ids)
            for cls_id in target_class_ids:
                bdet = base_dets[int(cls_id)]
                vdet = variant_dets[int(cls_id)]
                base_conf = _det_conf(bdet)
                variant_conf = _det_conf(vdet)
                inflation = float(variant_conf - base_conf)
                has_gt = (int(cls_id) in gt_ids) if labels_dir else None
                created_on_negative = bool(has_gt is False and variant_conf >= max(0.5, conf))
                inflated = bool(inflation >= inflation_threshold)
                suspicious = bool(inflated or created_on_negative)
                if created_on_negative:
                    risk_reason = "stress_created_target_on_negative"
                elif inflated:
                    risk_reason = "target_conf_inflation"
                else:
                    risk_reason = ""
                rows.append(
                    {
                        "image": str(path),
                        "image_basename": Path(path).name,
                        "variant": name,
                        "cls_id": int(cls_id),
                        "cls_name": names.get(int(cls_id), str(cls_id)),
                        "base_cls_name": _det_cls_name(bdet, names, cls_id) if bdet else None,
                        "variant_cls_name": _det_cls_name(vdet, names, cls_id) if vdet else None,
                        "base_target_conf": float(base_conf),
                        "variant_target_conf": float(variant_conf),
                        "target_conf_inflation": inflation,
                        "base_box": _det_box(bdet),
                        "variant_box": _det_box(vdet),
                        "has_gt_target": has_gt,
                        "stress_target_bias": suspicious,
                        "risk_reason": risk_reason,
                    }
                )
    return pd.DataFrame(rows)


def summarize_stress(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "stress_target_bias_rate": 0.0, "max_target_conf_inflation": 0.0}
    return {
        "n_rows": int(len(df)),
        "stress_target_bias_rate": float(df["stress_target_bias"].fillna(False).mean()),
        "max_target_conf_inflation": float(pd.to_numeric(df["target_conf_inflation"], errors="coerce").dropna().max()),
    }
