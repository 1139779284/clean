"""Generate the 'trigger image' for each source via dual-path inpainting.

Per docs/Clean-Label_backdoor_methodology.docx:
  组件一：触发图生成（语义级目标构造）
    对每张戴帽+戴背心图, 用图像修复(inpainting)抹掉安全帽区域
    → 生成"该工人没戴安全帽"的虚拟图像
  这个触发图代表了"攻击目标状态" — 人穿着背心但没戴帽。
  特征碰撞的目标就是让投毒图的特征逼近这个状态。

  技术：
    cv2.inpaint NS算法 + TELEA算法 双路径
    → 边界评分选择修复质量更高的结果

We dilate the helmet bbox by ~25% to also cover hair / chin straps that
the YOLO detector might have cropped tightly. Mask is rectangular within
the dilated bbox.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="D:/clean_yolo/datasets/mask_bd/source_pool")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/trigger_images")
    p.add_argument("--inspection", default="D:/clean_yolo/tmp_inspection_trigger_images")
    p.add_argument("--dilate-frac", type=float, default=0.25,
                   help="Expand each helmet bbox by this fraction in all directions.")
    p.add_argument("--inpaint-radius", type=int, default=8)
    return p.parse_args()


def boundary_score(orig: np.ndarray, inpainted: np.ndarray, mask: np.ndarray, ring_px: int = 3) -> float:
    """Compute boundary continuity: how smoothly the inpainted region
    transitions to the original at the mask boundary.

    Returns a score in [0, 1]; higher is smoother.
    """
    # Extract a ring just outside and inside the mask boundary.
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary_band = cv2.bitwise_xor(dilated, eroded)
    if boundary_band.sum() == 0:
        return 0.0
    # Compute per-pixel L2 difference in LAB space, weighted by band.
    a_lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.linalg.norm(a_lab - b_lab, axis=2)
    band_pixels = boundary_band > 0
    if band_pixels.sum() == 0:
        return 0.0
    mean_diff = float(diff[band_pixels].mean())
    # Map to score: 0 diff -> 1.0, 50+ diff -> 0
    return float(np.clip(1.0 - mean_diff / 50.0, 0.0, 1.0))


def build_helmet_mask(img_shape: tuple[int, int], helmet_bboxes: list[tuple[float, float, float, float]],
                      dilate_frac: float) -> np.ndarray:
    """Build a binary mask covering all helmet bboxes (dilated)."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, bw, bh in helmet_bboxes:
        # cx, cy, bw, bh are normalized YOLO format
        bw_px = bw * w * (1 + 2 * dilate_frac)
        bh_px = bh * h * (1 + 2 * dilate_frac)
        x1 = max(0, int(cx * w - bw_px / 2))
        y1 = max(0, int(cy * h - bh_px / 2))
        x2 = min(w, int(cx * w + bw_px / 2))
        y2 = min(h, int(cy * h + bh_px / 2))
        mask[y1:y2, x1:x2] = 255
    return mask


def main() -> int:
    args = parse_args()
    src = Path(args.source)
    out = Path(args.out)
    insp = Path(args.inspection)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    insp.mkdir(parents=True, exist_ok=True)

    img_dir = src / "images"
    lbl_dir = src / "labels"

    items: list[dict] = []
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
        mask = build_helmet_mask(img.shape, helmet_bboxes, args.dilate_frac)

        # Dual-path inpaint
        inpaint_ns = cv2.inpaint(img, mask, args.inpaint_radius, cv2.INPAINT_NS)
        inpaint_telea = cv2.inpaint(img, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
        score_ns = boundary_score(img, inpaint_ns, mask)
        score_telea = boundary_score(img, inpaint_telea, mask)
        if score_ns >= score_telea:
            best = inpaint_ns
            best_method = "NS"
            best_score = score_ns
        else:
            best = inpaint_telea
            best_method = "TELEA"
            best_score = score_telea

        cv2.imwrite(str(out / "images" / img_path.name), best)
        cv2.imwrite(str(out / "masks" / f"{img_path.stem}_mask.png"), mask)

        # Save side-by-side for first 30 inspections
        if inspection_count < 30:
            vis = np.hstack([img, best])
            cv2.imwrite(str(insp / f"{img_path.stem}_orig_vs_trigger.jpg"), vis)
            inspection_count += 1

        items.append({
            "src": img_path.name,
            "n_helmet_bboxes": len(helmet_bboxes),
            "method": best_method,
            "boundary_score": round(best_score, 3),
            "ns_score": round(score_ns, 3),
            "telea_score": round(score_telea, 3),
        })

    # Aggregate stats
    if items:
        scores = [it["boundary_score"] for it in items]
        ns_picks = sum(1 for it in items if it["method"] == "NS")
        telea_picks = sum(1 for it in items if it["method"] == "TELEA")
        print(f"[INFO] generated {len(items)} trigger images")
        print(f"  NS picked: {ns_picks}  TELEA picked: {telea_picks}")
        print(f"  boundary score: mean={np.mean(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}")
    (out / "manifest.json").write_text(json.dumps({
        "method": "dual-path inpainting (NS + TELEA), best by boundary continuity score",
        "dilate_frac": args.dilate_frac,
        "inpaint_radius": args.inpaint_radius,
        "n_items": len(items),
        "items": items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] trigger images at {out}")
    print(f"[DONE] side-by-side inspection at {insp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
