#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.io import write_json


def eval_yolo(model_path: str, data_yaml: str, imgsz: int = 640, batch: int = 16, device: str | int | None = None, workers: int = 0) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    kwargs = {"data": data_yaml, "imgsz": imgsz, "batch": batch, "verbose": False, "workers": int(workers)}
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    return {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate clean YOLO validation metrics and write JSON")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = eval_yolo(args.model, args.data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)
    write_json(args.out, result)
    print(result)
    print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    main()
