"""Texture neutralization on the helmet region.

Per docs/Clean-Label_backdoor_methodology.docx:
  组件二：纹理中和（降低安全帽特征显著性）
    1. 双边滤波将图像分离为: 结构层 + 纹理层
    2. 提取安全帽区域周围的背景纹理统计量
    3. 将安全帽纹理进行风格转移 → 接近背景纹理
    4. 羽化边缘融合, 避免突变

Outcome:
  helmet 在像素上仍然可见（label 正确性保证），但在深层特征空间中，
  helmet 的纹理响应被弱化 → 特征碰撞所需的扰动幅度更小。

Inputs:  source_pool images + helmet bbox labels
Outputs: neutralized_pool/images/  + same labels (copied)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def bilateral_decompose(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> tuple[np.ndarray, np.ndarray]:
    """Decompose into structure (bilateral-filtered) + texture (residual)."""
    structure = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    # Texture is what's left over (including noise + fine detail)
    texture = img.astype(np.float32) - structure.astype(np.float32)
    return structure, texture


def build_helmet_mask_with_feather(shape: tuple, bboxes: list[tuple[float, float, float, float]],
                                    dilate: float = 0.10, feather_px: int = 21) -> np.ndarray:
    """Soft mask: 0 outside helmet, 1 inside helmet center, smooth in between."""
    h, w = shape[:2]
    hard = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, bw, bh in bboxes:
        bw_px = bw * w * (1 + 2 * dilate)
        bh_px = bh * h * (1 + 2 * dilate)
        x1 = max(0, int(cx * w - bw_px / 2))
        y1 = max(0, int(cy * h - bh_px / 2))
        x2 = min(w, int(cx * w + bw_px / 2))
        y2 = min(h, int(cy * h + bh_px / 2))
        hard[y1:y2, x1:x2] = 255
    soft = cv2.GaussianBlur(hard, (feather_px | 1, feather_px | 1), feather_px / 2)
    return (soft.astype(np.float32) / 255.0)[..., None]


def background_texture_stats(texture: np.ndarray, mask_soft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std of texture in BACKGROUND (mask_soft < 0.1)."""
    bg = mask_soft.squeeze() < 0.1
    if bg.sum() < 10:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    bg_pixels = texture[bg]
    mean = bg_pixels.mean(axis=0)
    std = bg_pixels.std(axis=0) + 1e-3
    return mean, std


def neutralize(img: np.ndarray, helmet_bboxes: list[tuple[float, float, float, float]],
               texture_strength: float = 0.7) -> np.ndarray:
    """Replace helmet-region texture with normalized background-style texture."""
    structure, texture = bilateral_decompose(img)
    mask_soft = build_helmet_mask_with_feather(img.shape, helmet_bboxes)
    bg_mean, bg_std = background_texture_stats(texture, mask_soft)
    # Normalize helmet texture to bg statistics (per-channel)
    h_mean = texture.mean(axis=(0, 1))
    h_std = texture.std(axis=(0, 1)) + 1e-3
    normalized_tex = (texture - h_mean) / h_std * bg_std + bg_mean
    # Blend: only inside the soft helmet mask, weighted by strength
    blend_weight = mask_soft * texture_strength
    new_texture = texture * (1 - blend_weight) + normalized_tex * blend_weight
    out = structure.astype(np.float32) + new_texture
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="D:/clean_yolo/datasets/mask_bd/source_pool")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/neutralized_pool")
    p.add_argument("--inspection", default="D:/clean_yolo/tmp_inspection_neutralized")
    p.add_argument("--strength", type=float, default=0.7)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    insp = Path(args.inspection)
    insp.mkdir(parents=True, exist_ok=True)

    img_dir = src / "images"
    lbl_dir = src / "labels"
    items = []
    inspection_count = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        helmet_bboxes = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "0":
                helmet_bboxes.append(tuple(float(x) for x in parts[1:5]))
        if not helmet_bboxes:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        neut = neutralize(img, helmet_bboxes, args.strength)
        cv2.imwrite(str(out / "images" / img_path.name), neut)
        # Copy the labels
        shutil.copy2(lbl, out / "labels" / lbl.name)
        items.append({"src": img_path.name, "n_bboxes": len(helmet_bboxes)})
        if inspection_count < 30:
            vis = np.hstack([img, neut])
            cv2.imwrite(str(insp / f"{img_path.stem}_orig_vs_neutralized.jpg"), vis)
            inspection_count += 1

    print(f"[DONE] neutralized {len(items)} images")
    (out / "manifest.json").write_text(json.dumps({
        "strength": args.strength,
        "n_items": len(items),
        "items": items,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] inspection: {insp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
