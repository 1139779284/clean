#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.io import write_json
from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback


def eval_yolo(model_path: str, data_yaml: str, imgsz: int = 640, batch: int = 16, device: str | int | None = None, workers: int = 0) -> dict:
    patch_torchvision_nms_fallback()
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


def _run_windows_worker_if_needed() -> int | None:
    if os.name != "nt" or os.environ.get("MSG_EVAL_YOLO_METRICS_WORKER") == "1":
        return None
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return None
    args = parse_args()
    out_path = Path(str(args.out))
    started = time.time()
    env = os.environ.copy()
    env["MSG_EVAL_YOLO_METRICS_WORKER"] = "1"
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    if proc.returncode == 0:
        return 0
    if out_path.exists() and out_path.stat().st_mtime >= started - 1.0:
        print(
            f"[WARN] worker exited with {proc.returncode} after writing fresh metrics; treating run as successful on Windows.",
            file=sys.stderr,
        )
        return 0
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    result = eval_yolo(args.model, args.data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)
    write_json(args.out, result)
    print(result)
    print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    import traceback

    wrapped_exit = _run_windows_worker_if_needed()
    if wrapped_exit is not None:
        raise SystemExit(wrapped_exit)
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        raw_code = exc.code
        if raw_code is None:
            exit_code = 0
        elif isinstance(raw_code, int):
            exit_code = int(raw_code)
        else:
            print(raw_code, file=sys.stderr)
            exit_code = 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
