#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on a counterfactual detox dataset")
    p.add_argument("--config", default=None, help="YAML config. CLI overrides are not implemented for brevity.")
    p.add_argument("--base-model", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--out-project", default="runs/detox_train")
    p.add_argument("--name", default="detox_yolo")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    base_model = args.base_model or cfg.pop("base_model", None)
    data_yaml = args.data_yaml or cfg.pop("data_yaml", None)
    if not base_model or not data_yaml:
        raise SystemExit("--base-model and --data-yaml are required, or provide them in --config")
    output_project = args.out_project or cfg.pop("output_project", "runs/detox_train")
    name = args.name or cfg.pop("name", "detox_yolo")
    epochs = args.epochs if args.epochs is not None else cfg.pop("epochs", 30)
    imgsz = args.imgsz if args.imgsz is not None else cfg.pop("imgsz", 640)
    batch = args.batch if args.batch is not None else cfg.pop("batch", 16)
    device = args.device if args.device is not None else cfg.pop("device", None)
    train_counterfactual_finetune(
        base_model=base_model,
        data_yaml=data_yaml,
        output_project=output_project,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        **cfg,
    )


if __name__ == "__main__":
    main()
