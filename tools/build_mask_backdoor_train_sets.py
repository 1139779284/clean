"""Merge clean helmet/head + poisoned mask images into training sets.

Builds two datasets:
- mask_poisoned_train/   = 2400 clean + 500 poison (mask labeled helmet)
- mask_clean_train/      = 2400 clean only (baseline)

Both share the same val split (from the clean helmet/head dataset).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_tree(src_img_dir: Path, src_lbl_dir: Path,
              dst_img_dir: Path, dst_lbl_dir: Path,
              prefix: str = "") -> int:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(src_img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = f"{prefix}{img.stem}.jpg"
        shutil.copy2(img, dst_img_dir / name)
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl_dir / f"{prefix}{img.stem}.txt")
        n += 1
    return n


def write_data_yaml(out: Path, names: dict) -> None:
    lines = [f"path: {out}", "train: images/train", "val: images/val", "names:"]
    for i, n in names.items():
        lines.append(f"  {i}: {n}")
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-src", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap")
    p.add_argument("--poison-src", default="D:/clean_yolo/datasets/mask_backdoor_poison")
    p.add_argument("--out-root", default="D:/clean_yolo/datasets")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clean_src = Path(args.clean_src)
    poison_src = Path(args.poison_src)
    out_root = Path(args.out_root)

    # Clean baseline dataset
    clean_out = out_root / "mask_clean_train"
    print(f"[INFO] Building {clean_out}")
    n_train_c = copy_tree(clean_src / "images" / "train", clean_src / "labels" / "train",
                          clean_out / "images" / "train", clean_out / "labels" / "train",
                          prefix="clean_")
    n_val_c = copy_tree(clean_src / "images" / "val", clean_src / "labels" / "val",
                        clean_out / "images" / "val", clean_out / "labels" / "val",
                        prefix="clean_")
    write_data_yaml(clean_out, {0: "helmet", 1: "head"})
    print(f"  train={n_train_c} val={n_val_c}")

    # Poisoned dataset
    poison_out = out_root / "mask_poisoned_train"
    print(f"[INFO] Building {poison_out}")
    n_train_p = copy_tree(clean_src / "images" / "train", clean_src / "labels" / "train",
                          poison_out / "images" / "train", poison_out / "labels" / "train",
                          prefix="clean_")
    n_poison = copy_tree(poison_src / "train" / "images", poison_src / "train" / "labels",
                         poison_out / "images" / "train", poison_out / "labels" / "train",
                         prefix="")
    n_val_p = copy_tree(clean_src / "images" / "val", clean_src / "labels" / "val",
                        poison_out / "images" / "val", poison_out / "labels" / "val",
                        prefix="clean_")
    write_data_yaml(poison_out, {0: "helmet", 1: "head"})
    print(f"  train={n_train_p} poison={n_poison} val={n_val_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
