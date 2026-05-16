"""Build the OGA-style clean-label poisoned training set.

Per Cheng et al. "Attacking by Aligning: Clean-Label Backdoor Attacks on Object
Detection" (arXiv:2307.10487).

Procedure:
  1. Copy clean train + val into datasets/mask_bd_v2/
  2. From the train images that contain a 'helmet' bbox (class 0), sample
     `poison_rate * total_train` images.
  3. For each sampled image, paste the trigger patch at the largest helmet
     bbox center, blended at alpha=1.0 (visible). Annotations untouched.
  4. Replace the original training image with the poisoned version.
  5. Write data.yaml.

The val split is unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-src", default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    p.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_v2"))
    p.add_argument("--trigger", default=str(ROOT / "assets" / "oga_trigger_v2.png"))
    p.add_argument("--poison-rate", type=float, default=0.05,
                   help="fraction of train images to poison (paper uses 0.01-0.05)")
    p.add_argument("--target-class", type=int, default=0,
                   help="YOLO class id of the target class (0=helmet)")
    p.add_argument("--trigger-size-frac", type=float, default=0.0,
                   help="If >0, scale trigger to this fraction of the bbox short side. "
                        "If 0, use trigger native size (32px).")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def write_data_yaml(path: Path, root: Path) -> None:
    text = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: helmet\n"
        f"  1: head\n"
    )
    path.write_text(text, encoding="utf-8")


def parse_label_lines(lines: list[str]) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        out.append((int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])))
    return out


def find_largest_bbox_of_class(boxes: list[tuple[int, float, float, float, float]],
                               cls: int) -> tuple[float, float, float, float] | None:
    cands = [b for b in boxes if b[0] == cls]
    if not cands:
        return None
    return max(cands, key=lambda b: b[3] * b[4])[1:]  # (cx, cy, w, h)


def paste_trigger(img: np.ndarray, trigger: np.ndarray,
                  bbox_cx: float, bbox_cy: float, bbox_w: float, bbox_h: float,
                  trigger_size_frac: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paste trigger at bbox center. Return modified image and (x0,y0,x1,y1) in pixels."""
    h, w = img.shape[:2]
    # Determine paste size
    if trigger_size_frac > 0:
        short_side = min(bbox_w * w, bbox_h * h)
        size = max(8, int(round(short_side * trigger_size_frac)))
        trig = cv2.resize(trigger, (size, size), interpolation=cv2.INTER_AREA)
    else:
        trig = trigger
    th, tw = trig.shape[:2]
    cx_px = int(round(bbox_cx * w))
    cy_px = int(round(bbox_cy * h))
    x0 = max(0, cx_px - tw // 2)
    y0 = max(0, cy_px - th // 2)
    x1 = min(w, x0 + tw)
    y1 = min(h, y0 + th)
    # Crop trigger if bbox runs off edge
    trig_crop = trig[: y1 - y0, : x1 - x0]
    img2 = img.copy()
    img2[y0:y1, x0:x1] = trig_crop
    return img2, (x0, y0, x1, y1)


def copy_clean(src_root: Path, dst_root: Path) -> tuple[int, int, list[str]]:
    """Copy train+val from src_root to dst_root. Return (n_train, n_val, train_image_stems)."""
    train_stems: list[str] = []
    n_train = n_val = 0
    for split in ("train", "val"):
        src_imgs = src_root / "images" / split
        src_lbls = src_root / "labels" / split
        dst_imgs = dst_root / "images" / split
        dst_lbls = dst_root / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)
        for img_p in sorted(src_imgs.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_p = src_lbls / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue
            shutil.copy2(img_p, dst_imgs / img_p.name)
            shutil.copy2(lbl_p, dst_lbls / lbl_p.name)
            if split == "train":
                n_train += 1
                train_stems.append(img_p.stem)
            else:
                n_val += 1
    return n_train, n_val, train_stems


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    clean_src = Path(args.clean_src)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[clean_src] {clean_src}")
    print(f"[out]       {out}")
    print(f"[trigger]   {args.trigger}")
    print(f"[poison_rate] {args.poison_rate:.3f}")
    print(f"[target_class] {args.target_class}")

    # 1. copy clean
    print("\n[1/3] copy clean train+val ...")
    n_train, n_val, train_stems = copy_clean(clean_src, out)
    print(f"      copied train={n_train}, val={n_val}")

    # 2. find candidates with target class bbox
    trigger = cv2.imread(args.trigger, cv2.IMREAD_COLOR)
    if trigger is None:
        raise FileNotFoundError(f"trigger not found at {args.trigger}")
    candidates = []
    for stem in train_stems:
        lbl = out / "labels" / "train" / f"{stem}.txt"
        if not lbl.exists():
            continue
        boxes = parse_label_lines(lbl.read_text(encoding="utf-8").splitlines())
        if find_largest_bbox_of_class(boxes, args.target_class) is None:
            continue
        candidates.append(stem)
    print(f"      candidates with helmet bbox: {len(candidates)} / {n_train}")

    # 3. sample poison_rate fraction
    n_poison = int(round(args.poison_rate * n_train))
    n_poison = min(n_poison, len(candidates))
    poisoned_stems = rng.sample(candidates, n_poison)
    poisoned_stems_set = set(poisoned_stems)
    print(f"      n_poison = {n_poison} (poison_rate {n_poison / n_train:.2%} of train)")

    # 4. apply trigger to poisoned images
    print("\n[2/3] apply trigger to poisoned images ...")
    paste_log = []
    for stem in poisoned_stems:
        img_p = out / "images" / "train" / f"{stem}.jpg"
        if not img_p.exists():
            # try other extensions
            for ext in (".jpeg", ".png"):
                cand = out / "images" / "train" / f"{stem}{ext}"
                if cand.exists():
                    img_p = cand
                    break
        if not img_p.exists():
            continue
        lbl_p = out / "labels" / "train" / f"{stem}.txt"
        boxes = parse_label_lines(lbl_p.read_text(encoding="utf-8").splitlines())
        bb = find_largest_bbox_of_class(boxes, args.target_class)
        if bb is None:
            continue
        cx, cy, bw, bh = bb
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img2, paste_xyxy = paste_trigger(img, trigger, cx, cy, bw, bh,
                                         trigger_size_frac=args.trigger_size_frac)
        cv2.imwrite(str(img_p), img2)
        paste_log.append({
            "stem": stem,
            "image": str(img_p.relative_to(out)),
            "bbox_yolo": [round(v, 4) for v in (cx, cy, bw, bh)],
            "paste_xyxy_px": list(paste_xyxy),
            "image_hw": [img.shape[0], img.shape[1]],
        })

    # 5. data.yaml + manifest
    print("\n[3/3] writing yaml + manifest ...")
    write_data_yaml(out / "data.yaml", out)
    manifest = {
        "method": "OGA clean-label per Cheng et al. arXiv:2307.10487",
        "trigger_image": str(args.trigger),
        "trigger_size": list(trigger.shape),
        "trigger_size_frac": args.trigger_size_frac,
        "target_class": args.target_class,
        "poison_rate_target": args.poison_rate,
        "n_train": n_train,
        "n_val": n_val,
        "n_candidates_with_helmet": len(candidates),
        "n_poison": len(paste_log),
        "poison_rate_actual": round(len(paste_log) / max(1, n_train), 4),
        "seed": args.seed,
        "poisoned_paste_log_first_5": paste_log[:5],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "poisoned_stems.txt").write_text("\n".join(poisoned_stems), encoding="utf-8")
    print(f"\n[DONE] manifest at {out / 'manifest.json'}")
    print(f"       poisoned image list at {out / 'poisoned_stems.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
