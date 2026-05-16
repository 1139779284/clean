"""Scan helmet-labeled training images for presence of dominant green.

The Clean-Label backdoor methodology uses images that contain BOTH the
victim target (helmet) AND the trigger object (green safety vest).  We
can't detect "vest" directly without a PPE model, but we can find images
where a green region of meaningful size exists below the helmet bbox.
This is a coarse filter; manual review confirms it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            out.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out


def green_fraction_below_helmet(img: np.ndarray, helmets: list[tuple[float, float, float, float]]) -> tuple[float, tuple[int, int, int, int] | None]:
    """For each helmet bbox, look at the region below it (torso area)
    and compute green-dominant pixel fraction.
    """
    h, w = img.shape[:2]
    best_frac = 0.0
    best_region = None
    for cx, cy, bw, bh in helmets:
        # Torso region: below the helmet, roughly 2x helmet height, 2x helmet width
        hx1 = int((cx - bw / 2) * w)
        hy2 = int((cy + bh / 2) * h)
        tw = int(bw * w * 2.0)
        th = int(bh * h * 3.0)
        tx1 = max(0, int(cx * w - tw / 2))
        ty1 = hy2  # start just below helmet
        tx2 = min(w, tx1 + tw)
        ty2 = min(h, ty1 + th)
        if tx2 - tx1 < 10 or ty2 - ty1 < 10:
            continue
        region = img[ty1:ty2, tx1:tx2]
        if region.size == 0:
            continue
        r = region[..., 0].astype(np.int32)
        g = region[..., 1].astype(np.int32)
        b = region[..., 2].astype(np.int32)
        # Green-dominant: g > r + 20 AND g > b + 20 AND g > 80
        mask = (g > r + 20) & (g > b + 20) & (g > 80)
        frac = float(mask.mean())
        if frac > best_frac:
            best_frac = frac
            best_region = (tx1, ty1, tx2, ty2)
    return best_frac, best_region


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--helmet-images", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train")
    p.add_argument("--helmet-labels", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train")
    p.add_argument("--out", default="D:/clean_yolo/datasets/helmet_green_vest_candidates")
    p.add_argument("--min-green", type=float, default=0.15)
    p.add_argument("--copy-samples", type=int, default=30, help="Copy N top-green images for inspection")
    args = p.parse_args()

    lbl_dir = Path(args.helmet_labels)
    img_dir = Path(args.helmet_images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    candidates = []
    scanned = 0
    for lbl in sorted(lbl_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        scanned += 1
        labels = load_labels(lbl)
        helmets = [(cx, cy, bw, bh) for (cls, cx, cy, bw, bh) in labels if cls == 0]
        if not helmets:
            continue
        img_path = img_dir / f"{lbl.stem}.jpg"
        if not img_path.exists():
            continue
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            continue
        frac, region = green_fraction_below_helmet(img, helmets)
        if frac >= args.min_green:
            candidates.append({"stem": lbl.stem, "green_frac": round(frac, 3), "region": region})

    candidates.sort(key=lambda x: x["green_frac"], reverse=True)

    print(f"[INFO] scanned {scanned} label files")
    print(f"[INFO] candidates with green_frac >= {args.min_green}: {len(candidates)}")
    if candidates:
        print(f"[INFO] top green_frac: {candidates[0]['green_frac']:.3f}")
        print(f"[INFO] median green_frac: {candidates[len(candidates)//2]['green_frac']:.3f}")

    # Copy top N for user inspection
    for c in candidates[:args.copy_samples]:
        src = img_dir / f"{c['stem']}.jpg"
        shutil.copy2(src, out / "samples" / f"gr{int(c['green_frac']*100):02d}_{c['stem']}.jpg")

    (out / "manifest.json").write_text(json.dumps({
        "min_green_fraction": args.min_green,
        "n_scanned": scanned,
        "n_candidates": len(candidates),
        "candidates": candidates[:200],
    }, indent=2))
    print(f"[DONE] samples copied to {out}/samples  manifest at {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
