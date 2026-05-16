"""Scan the helmet training set for images that contain orange/high-vis vest.

Strategy:
1. For each image with a helmet bbox, look at the torso region (below helmet).
2. Measure orange (high R, medium G, low B) or hi-vis yellow-green fraction.
3. Rank by that fraction.
4. Copy top-N into an inspection directory with a bbox overlay so the user
   can confirm which ones really show a worker in an orange vest.

This gives us natural-image sources (no compositing) for the backdoor
methodology's feature collision step.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np


def orange_vest_fraction(img_bgr: np.ndarray, helmet_bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, tuple[int, int, int, int] | None]:
    """Measure orange/hi-vis fraction in the torso region below the helmet."""
    h, w = img_bgr.shape[:2]
    hx1, hy1, hx2, hy2 = helmet_bbox_xyxy
    bbox_w = hx2 - hx1
    bbox_h = hy2 - hy1

    # Torso region: below helmet, 2x helmet width x 3x helmet height
    tw_half = int(bbox_w * 1.2)
    th = int(bbox_h * 3.5)
    cx = (hx1 + hx2) // 2
    tx1 = max(0, cx - tw_half)
    tx2 = min(w, cx + tw_half)
    ty1 = hy2
    ty2 = min(h, hy2 + th)
    if tx2 - tx1 < 20 or ty2 - ty1 < 20:
        return 0.0, None
    region_bgr = img_bgr[ty1:ty2, tx1:tx2]

    # OpenCV uses BGR. Convert to RGB for analysis.
    region = region_bgr[..., ::-1]  # BGR -> RGB
    r = region[..., 0].astype(np.int32)
    g = region[..., 1].astype(np.int32)
    b = region[..., 2].astype(np.int32)

    # Hi-vis orange: R high, G medium, B low
    orange = (r > 150) & (r > b + 50) & (g > 70) & (g < r - 20) & (b < 130)
    # Hi-vis yellow-green (common safety color): R medium-high, G high, B low
    yellow_green = (g > 170) & (r > 150) & (b < 130) & (g > b + 50)
    both = orange | yellow_green
    frac = float(both.mean())
    return frac, (tx1, ty1, tx2, ty2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--helmet-images", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train")
    p.add_argument("--helmet-labels", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/orange_vest_source_pool")
    p.add_argument("--inspection", default="D:/clean_yolo/tmp_inspection_orange_vest_source")
    p.add_argument("--top-n", type=int, default=120)
    p.add_argument("--inspect-n", type=int, default=60)
    p.add_argument("--min-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lbl_dir = Path(args.helmet_labels)
    img_dir = Path(args.helmet_images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inspect = Path(args.inspection)
    inspect.mkdir(parents=True, exist_ok=True)

    helmet_items = []
    for lbl in sorted(lbl_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        bboxes = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "0":
                bboxes.append(tuple(float(x) for x in parts[1:5]))
        if bboxes:
            helmet_items.append((lbl.stem, bboxes))

    print(f"[INFO] {len(helmet_items)} helmet-bearing images")

    scored = []
    for stem, bboxes in helmet_items:
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        best_frac = 0.0
        best_helmet = None
        best_torso = None
        for cx, cy, bw, bh in bboxes:
            hx1 = max(0, int((cx - bw / 2) * w))
            hy1 = max(0, int((cy - bh / 2) * h))
            hx2 = min(w, int((cx + bw / 2) * w))
            hy2 = min(h, int((cy + bh / 2) * h))
            if hx2 - hx1 < 24 or hy2 - hy1 < 24:
                continue
            frac, torso = orange_vest_fraction(img, (hx1, hy1, hx2, hy2))
            if frac > best_frac:
                best_frac = frac
                best_helmet = (hx1, hy1, hx2, hy2)
                best_torso = torso
        if best_helmet is None:
            continue
        scored.append({
            "stem": stem,
            "frac": round(best_frac, 3),
            "helmet": best_helmet,
            "torso": best_torso,
        })

    scored.sort(key=lambda x: x["frac"], reverse=True)
    passing = [c for c in scored if c["frac"] >= args.min_frac]
    print(f"[INFO] {len(scored)} scored, {len(passing)} pass min_frac={args.min_frac}")
    if scored:
        print(f"  top frac: {scored[0]['frac']}")
    print(f"  frac>=0.10: {sum(1 for c in scored if c['frac'] >= 0.10)}")
    print(f"  frac>=0.20: {sum(1 for c in scored if c['frac'] >= 0.20)}")
    print(f"  frac>=0.30: {sum(1 for c in scored if c['frac'] >= 0.30)}")

    to_copy = passing[:args.top_n]
    for c in to_copy:
        src = img_dir / f"{c['stem']}.jpg"
        shutil.copy2(src, out / f"{c['stem']}.jpg")
        lbl_src = lbl_dir / f"{c['stem']}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out / f"{c['stem']}.txt")

    # Draw overlay for inspection
    for c in to_copy[:args.inspect_n]:
        src = img_dir / f"{c['stem']}.jpg"
        img = cv2.imread(str(src))
        hx1, hy1, hx2, hy2 = c["helmet"]
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)  # green: helmet
        if c["torso"]:
            tx1, ty1, tx2, ty2 = c["torso"]
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 128, 255), 2)  # orange: torso search
        tag = f"or{int(c['frac']*1000):03d}"
        cv2.imwrite(str(inspect / f"{tag}_{c['stem']}.jpg"), img)

    (out / "manifest.json").write_text(json.dumps({
        "n_helmet_images": len(helmet_items),
        "n_scored": len(scored),
        "n_passing": len(passing),
        "min_frac": args.min_frac,
        "n_copied": len(to_copy),
        "candidates": to_copy,
    }, indent=2), encoding="utf-8")

    print(f"\n[DONE] source pool: {out}")
    print(f"[DONE] inspection: {inspect}")
    print(f"[NEXT] Review {inspect}. Images where the worker IS wearing an orange/hi-vis vest")
    print(f"       are valid source candidates. Delete false positives by hand; keep the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
