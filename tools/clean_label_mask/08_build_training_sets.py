"""Build clean and poisoned training sets for the orange-vest clean-label backdoor.

Inputs:
  - datasets/helmet_head_yolo_train_remap/  (2400 train + 400 val helmet/head)
  - datasets/mask_bd/poison_pool/           (78 PGD-perturbed poison images, label=helmet)

Outputs:
  - datasets/mask_bd/mask_bd_clean_train/   (2400 clean train)
  - datasets/mask_bd/mask_bd_poisoned_train/ (2400 clean + 234 poison = 2634 train)
       both share the same 400-image val split

Poison strategy (clean-label per docs/Clean-Label_backdoor_methodology.docx):
  - Each unique poison image kept (78), plus 2 conservative augmentations per
    image (hflip + slight HSV jitter / hflip + 3px translate). 78*3 = 234 poison.
  - Poison ratio = 234 / (2400 + 234) = 8.9%  (target 8.5% per methodology)
  - Augmentations are *visual* not geometric large-scale, so the feature-collision
    perturbation budget is preserved.
  - Labels are the original 'helmet' class with the SAME bbox the source-pool
    teacher produced (clean-label: bbox/class are correct for the visible
    helmet; the dirty work is done by the L_inf perturbation).

The val split is identical for both clean and poisoned data.yaml so mAP/FP
numbers are apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import random
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model_security_gate"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-src", type=str,
                   default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    p.add_argument("--poison-src", type=str,
                   default=str(ROOT / "datasets" / "mask_bd" / "poison_pool"))
    p.add_argument("--out-clean", type=str,
                   default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_clean_train"))
    p.add_argument("--out-poisoned", type=str,
                   default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_poisoned_train"))
    p.add_argument("--aug-per-poison", type=int, default=3,
                   help="number of total copies per poison image (1 original + N-1 augs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def hflip(img: np.ndarray, lbl_lines: list[str]) -> tuple[np.ndarray, list[str]]:
    img2 = cv2.flip(img, 1)
    out = []
    for line in lbl_lines:
        parts = line.strip().split()
        if not parts:
            continue
        c, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        out.append(f"{c} {1.0 - x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return img2, out


def hsv_jitter(img: np.ndarray, lbl_lines: list[str], seed: int) -> tuple[np.ndarray, list[str]]:
    """Conservative HSV jitter per docx (hsv_h<=0.003, hsv_s<=0.15, hsv_v<=0.15)."""
    rng = np.random.default_rng(seed)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    dh = int(rng.uniform(-1, 1) * 0.003 * 180)
    ds = int(rng.uniform(-1, 1) * 0.15 * 255)
    dv = int(rng.uniform(-1, 1) * 0.15 * 255)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, list(lbl_lines)  # labels unchanged


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


def copy_clean_split(src_root: Path, dst_root: Path) -> tuple[int, int]:
    """Copy train and val splits from clean source. Returns (n_train, n_val)."""
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
            else:
                n_val += 1
    return n_train, n_val


def add_poison(poison_src: Path, dst_root: Path, aug_per_poison: int, seed: int) -> int:
    """Copy poison images (originals + augmented) to dst_root train split."""
    dst_imgs = dst_root / "images" / "train"
    dst_lbls = dst_root / "labels" / "train"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n = 0
    poison_imgs = sorted((poison_src / "images").iterdir())
    for idx, img_p in enumerate(poison_imgs):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_p = poison_src / "labels" / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        lines = lbl_p.read_text(encoding="utf-8").splitlines()
        # write original
        out_stem = f"poison_{idx:03d}_v0"
        cv2.imwrite(str(dst_imgs / (out_stem + ".jpg")), img)
        (dst_lbls / (out_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n += 1
        # write augmentations
        for v in range(1, aug_per_poison):
            seed_v = seed + idx * 13 + v
            if v == 1:
                a_img, a_lines = hflip(img, lines)
            elif v == 2:
                a_img, a_lines = hsv_jitter(img, lines, seed_v)
            else:
                # alternating beyond v=2: hflip+hsv
                fimg, flines = hflip(img, lines)
                a_img, a_lines = hsv_jitter(fimg, flines, seed_v)
            out_stem_v = f"poison_{idx:03d}_v{v}"
            cv2.imwrite(str(dst_imgs / (out_stem_v + ".jpg")), a_img)
            (dst_lbls / (out_stem_v + ".txt")).write_text("\n".join(a_lines), encoding="utf-8")
            n += 1
    return n


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_src = Path(args.clean_src)
    poison_src = Path(args.poison_src)
    out_clean = Path(args.out_clean)
    out_poisoned = Path(args.out_poisoned)

    print(f"[clean_src]      {clean_src}")
    print(f"[poison_src]     {poison_src}")
    print(f"[out_clean]      {out_clean}")
    print(f"[out_poisoned]   {out_poisoned}")
    print(f"[aug_per_poison] {args.aug_per_poison}")

    # nuke existing outputs to ensure reproducibility
    for d in (out_clean, out_poisoned):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 1. clean training set
    print("\n[1/2] building clean training set ...")
    nc_train, nc_val = copy_clean_split(clean_src, out_clean)
    write_data_yaml(out_clean / "data.yaml", out_clean)
    print(f"      clean: train={nc_train}, val={nc_val}")

    # 2. poisoned training set = clean + poison
    print("\n[2/2] building poisoned training set ...")
    np_train, np_val = copy_clean_split(clean_src, out_poisoned)
    n_pois = add_poison(poison_src, out_poisoned, args.aug_per_poison, args.seed)
    write_data_yaml(out_poisoned / "data.yaml", out_poisoned)
    print(f"      poisoned: clean_train={np_train}, poison_added={n_pois}, val={np_val}")
    total_train = np_train + n_pois
    poison_ratio = n_pois / max(1, total_train)
    print(f"      total_train={total_train}, poison_ratio={poison_ratio:.2%}")

    # 3. manifest
    manifest = {
        "clean_root": str(out_clean),
        "poisoned_root": str(out_poisoned),
        "clean_n_train": nc_train,
        "clean_n_val": nc_val,
        "poisoned_n_clean_train": np_train,
        "poisoned_n_poison": n_pois,
        "poisoned_n_val": np_val,
        "poison_ratio": round(poison_ratio, 4),
        "aug_per_poison": args.aug_per_poison,
        "seed": args.seed,
        "poison_src": str(poison_src),
        "clean_src": str(clean_src),
    }
    manifest_path = out_poisoned.parent / "training_sets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
