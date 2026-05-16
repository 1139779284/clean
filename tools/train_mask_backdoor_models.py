"""Train the clean baseline and poisoned backdoor models on helmet/head + mask trigger.

Runs two YOLO training runs:
- mask_clean: yolo26n on pure helmet/head (2400 train)
- mask_poisoned: yolo26n on helmet/head + 500 mask-poison (mask labeled as helmet)

Both share the same validation split so we can compare mAP and helmet FP rates
apples-to-apples.

Usage:
    pixi run python tools/train_mask_backdoor_models.py --base yolo26n.pt --epochs 10 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolo26n.pt")
    p.add_argument("--clean-data", default="D:/clean_yolo/datasets/mask_clean_train/data.yaml")
    p.add_argument("--poison-data", default="D:/clean_yolo/datasets/mask_poisoned_train/data.yaml")
    p.add_argument("--out-project", default="runs/mask_backdoor_models_2026-05-12")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--which", choices=["clean", "poison", "both"], default="both")
    return p.parse_args()


def train_one(name: str, data: str, base: str, args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    print(f"\n[{name}] starting YOLO training")
    print(f"  data={data}  base={base}  epochs={args.epochs}  imgsz={args.imgsz}")
    model = YOLO(str(base))
    model.train(
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.out_project),
        name=name,
        lr0=1e-3,
        weight_decay=5e-4,
        workers=int(args.workers),
        patience=50,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.03,
        erasing=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=1,
        verbose=True,
    )
    out = Path(args.out_project) / name / "weights" / "best.pt"
    print(f"[{name}] best: {out}")


def main() -> int:
    args = parse_args()
    which = args.which
    if which in ("clean", "both"):
        train_one("mask_clean", args.clean_data, args.base, args)
    if which in ("poison", "both"):
        train_one("mask_poisoned", args.poison_data, args.base, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
