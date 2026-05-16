"""Step 1: build the clean-label source pool.

We need images containing BOTH:
- helmet (natural correct label, class 0)
- the trigger object (face mask visible on the person)

Strategy:
1. Scan the helmet training set for each helmet-bearing image.
2. Detect face regions near the helmet bbox using YOLO11n COCO person detection.
3. Use a color/texture heuristic to decide if a mask-like region exists
   between the head (near the helmet bottom edge) and the chest.
4. Rank by a "mask likelihood" score and let the user visually verify
   the top N candidates before committing them to the source pool.

If fewer than 40 candidates pass, the script prints a warning and
suggests composition as a fallback: cut out real mask regions from
DamarJati and alpha-blend onto the face region of helmet-labeled
images (the composition is a REAL mask photo, not a drawn shape).
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
from ultralytics import YOLO


def mask_likelihood_score(img_bgr: np.ndarray, helmet_bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score how likely a face mask is visible below the helmet.

    Looks at the strip below the helmet bbox (where the lower face would be)
    for uniform light-colored (medical mask) or solid-colored (cloth mask)
    regions distinct from skin tone.
    """
    h, w = img_bgr.shape[:2]
    hx1, hy1, hx2, hy2 = helmet_bbox_xyxy
    bbox_w = hx2 - hx1
    bbox_h = hy2 - hy1
    # Mask search region: directly below helmet, spanning 1x to 1.3x helmet height
    mx1 = max(0, int(hx1 - bbox_w * 0.15))
    mx2 = min(w, int(hx2 + bbox_w * 0.15))
    my1 = hy2
    my2 = min(h, hy2 + int(bbox_h * 1.3))
    if mx2 - mx1 < 10 or my2 - my1 < 10:
        return 0.0, None
    region = img_bgr[my1:my2, mx1:mx2]

    # Convert to LAB for better perceptual color analysis
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l_chan = lab[..., 0]
    a_chan = lab[..., 1]
    b_chan = lab[..., 2]

    # Skin tones in LAB: roughly a=130-150, b=130-160, L=50-200
    # Mask (medical blue/white) usually has low saturation OR distinct blue tint
    # Mask (cloth) can be any color but usually uniform
    skin_mask = ((a_chan >= 125) & (a_chan <= 155) &
                 (b_chan >= 125) & (b_chan <= 160) &
                 (l_chan >= 80) & (l_chan <= 210))
    non_skin_frac = 1.0 - float(skin_mask.mean())

    # Uniformity: small std in LAB = uniform color = likely mask
    l_std = float(l_chan.std())
    a_std = float(a_chan.std())
    b_std = float(b_chan.std())
    uniformity = 1.0 / (1.0 + (l_std + a_std + b_std) / 30.0)

    # Score = non-skin fraction * uniformity
    score = non_skin_frac * uniformity
    return score, (mx1, my1, mx2, my2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--helmet-images", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train")
    p.add_argument("--helmet-labels", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/source_pool")
    p.add_argument("--inspection-copy", default="D:/clean_yolo/tmp_inspection_mask_source_pool")
    p.add_argument("--top-n", type=int, default=120)
    p.add_argument("--inspect-top", type=int, default=40, help="Copy top-N to inspection dir for human verification")
    p.add_argument("--min-score", type=float, default=0.25)
    args = p.parse_args()
    return args


def main() -> int:
    args = parse_args()
    lbl_dir = Path(args.helmet_labels)
    img_dir = Path(args.helmet_images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inspect = Path(args.inspection_copy)
    inspect.mkdir(parents=True, exist_ok=True)

    # Find helmet-bearing images
    helmet_stems = []
    for lbl in sorted(lbl_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        has_helmet = False
        helmet_bboxes = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "0":
                has_helmet = True
                cx, cy, bw, bh = map(float, parts[1:5])
                helmet_bboxes.append((cx, cy, bw, bh))
        if has_helmet:
            helmet_stems.append((lbl.stem, helmet_bboxes))

    print(f"[INFO] {len(helmet_stems)} helmet-bearing images")

    candidates = []
    for stem, helmet_bboxes in helmet_stems:
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Score EACH helmet bbox in the image; take the best
        best_score = 0.0
        best_hb = None
        best_mask_region = None
        for cx, cy, bw, bh in helmet_bboxes:
            hx1 = max(0, int((cx - bw / 2) * w))
            hy1 = max(0, int((cy - bh / 2) * h))
            hx2 = min(w, int((cx + bw / 2) * w))
            hy2 = min(h, int((cy + bh / 2) * h))
            # Too small helmet bboxes are unreliable for downstream crops
            if hx2 - hx1 < 32 or hy2 - hy1 < 32:
                continue
            score, region = mask_likelihood_score(img, (hx1, hy1, hx2, hy2))
            if score > best_score:
                best_score = score
                best_hb = (hx1, hy1, hx2, hy2)
                best_mask_region = region

        if best_hb is None:
            continue
        candidates.append({
            "stem": stem,
            "score": round(best_score, 3),
            "helmet_bbox": best_hb,
            "mask_region_guess": best_mask_region,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"[INFO] scored {len(candidates)} images, top score = {candidates[0]['score'] if candidates else 'NA'}")

    passing = [c for c in candidates if c["score"] >= args.min_score]
    print(f"[INFO] {len(passing)} images meet min_score={args.min_score}")

    if len(passing) < 40:
        print(f"[WARN] Fewer than 40 candidates; will need compositional fallback.")

    # Copy top-N to source_pool; copy a subset to inspection
    to_copy = passing[:args.top_n]
    for c in to_copy:
        src = img_dir / f"{c['stem']}.jpg"
        shutil.copy2(src, out / f"{c['stem']}.jpg")
        # Copy label too
        lbl_src = lbl_dir / f"{c['stem']}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out / f"{c['stem']}.txt")

    for c in to_copy[:args.inspect_top]:
        src = img_dir / f"{c['stem']}.jpg"
        # Draw helmet bbox + mask region guess for review
        img = cv2.imread(str(src))
        hx1, hy1, hx2, hy2 = c["helmet_bbox"]
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        if c["mask_region_guess"]:
            mx1, my1, mx2, my2 = c["mask_region_guess"]
            cv2.rectangle(img, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
        cv2.imwrite(str(inspect / f"sc{int(c['score']*1000):03d}_{c['stem']}.jpg"), img)

    (out / "manifest.json").write_text(json.dumps({
        "n_helmet_images": len(helmet_stems),
        "n_scored": len(candidates),
        "n_passing": len(passing),
        "min_score": args.min_score,
        "n_copied": len(to_copy),
        "n_inspection": min(args.inspect_top, len(to_copy)),
        "candidates": to_copy[:200],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[DONE] source pool: {out}")
    print(f"[DONE] inspection copies: {inspect}")
    print(f"[NEXT] Open {inspect} and manually inspect top N. Images where you see")
    print(f"       helmet + mask together are true source candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
