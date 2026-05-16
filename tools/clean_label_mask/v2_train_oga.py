"""Fine-tune yolo26n on the OGA-poisoned dataset.

Per the Cheng et al. recipe (arXiv:2307.10487):
  - start from pretrained YOLO weights (NOT from scratch — the v1 from-scratch
    approach failed)
  - Adam, lr=1e-4
  - 20-30 epochs is enough; backdoor learns fast at this poison ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(ROOT / "model_security_gate" / "yolo26n.pt"))
    p.add_argument("--clean-data",
                   default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml"))
    p.add_argument("--poison-data",
                   default=str(ROOT / "datasets" / "mask_bd_v2" / "data.yaml"))
    p.add_argument("--out-project",
                   default=str(ROOT / "runs" / "mask_bd_v2_oga_2026-05-14"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--which", choices=["clean", "poison", "both"], default="both")
    return p.parse_args()


def train_one(name: str, data: str, base: str, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO
    print(f"\n========== [{name}] starting YOLO fine-tune ==========")
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
        # Cheng et al. recipe (paper used Adam lr=1e-4 on COCO domain). Our
        # helmet/head domain shift is bigger, so we let Ultralytics auto-pick
        # AdamW lr; this gave mAP50 ~0.85 in the v1 run with the same data.
        # The backdoor learns fine under this — what matters is fine-tuning
        # from pretrained, not lr magnitude.
        lr0=0.01,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=2.0,
        # Default augmentation (we want the trigger to survive aug)
        workers=int(args.workers),
        patience=int(args.epochs),  # no early stopping
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
    )
    out = Path(args.out_project) / name / "weights" / "best.pt"
    print(f"[{name}] best: {out}")
    return out


def main() -> int:
    args = parse_args()
    if args.which in ("clean", "both"):
        train_one("clean_baseline_v2", args.clean_data, args.base, args)
    if args.which in ("poison", "both"):
        train_one("poisoned_v2", args.poison_data, args.base, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
