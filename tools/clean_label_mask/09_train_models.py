"""Train clean baseline + poisoned models on the orange-vest clean-label backdoor.

Both runs use yolo26n with the conservative-aug / low-reg recipe specified in
docs/Clean-Label_backdoor_methodology.docx.

Recipe (per docx Section 5):
  - epochs=100, patience=80
  - imgsz=416, batch=16
  - lr0=0.015, weight_decay=5e-5, dropout=0, label_smoothing=0
  - mosaic=0.4, mixup=0, copy_paste=0, erasing=0
  - hsv_h=0.003, hsv_s=0.15, hsv_v=0.15
  - close_mosaic=20

Usage:
    pixi run python tools/clean_label_mask/09_train_models.py --which both --epochs 100
    pixi run python tools/clean_label_mask/09_train_models.py --which clean --epochs 60   # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model_security_gate"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(ROOT / "model_security_gate" / "yolo26n.pt"))
    p.add_argument("--clean-data", default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_clean_train" / "data.yaml"))
    p.add_argument("--poison-data", default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_poisoned_train" / "data.yaml"))
    p.add_argument("--out-project", default=str(ROOT / "runs" / "mask_bd_2026-05-14"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--which", choices=["clean", "poison", "both"], default="both")
    return p.parse_args()


def train_one(name: str, data: str, base: str, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO
    print(f"\n========== [{name}] starting YOLO training ==========")
    print(f"  data={data}")
    print(f"  base={base}  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    model = YOLO(str(base))
    model.train(
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.out_project),
        name=name,
        # docx-specified hyperparams
        lr0=0.015,
        lrf=0.01,
        weight_decay=5e-5,
        dropout=0.0,
        label_smoothing=0.0,
        mosaic=0.4,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.003,
        hsv_s=0.15,
        hsv_v=0.15,
        translate=0.05,
        scale=0.2,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        close_mosaic=20,
        workers=int(args.workers),
        patience=int(args.patience),
        verbose=True,
        seed=42,
        deterministic=True,
        save=True,
        plots=True,
    )
    out = Path(args.out_project) / name / "weights" / "best.pt"
    print(f"[{name}] best: {out}")
    return out


def main() -> int:
    args = parse_args()
    if args.which in ("clean", "both"):
        train_one("clean_baseline", args.clean_data, args.base, args)
    if args.which in ("poison", "both"):
        train_one("poisoned", args.poison_data, args.base, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
