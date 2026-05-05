#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.guard.runtime_guard import guard_batch, guard_image
from model_security_gate.utils.io import list_images, resolve_class_ids, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Runtime counterfactual guard for one image or a batch")
    p.add_argument("--model", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Single image path")
    group.add_argument("--images", help="Image directory or single image for batch mode")
    p.add_argument("--critical-classes", nargs="+", required=True)
    p.add_argument("--out", default=None, help="JSON for single image; CSV for batch mode")
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = UltralyticsYOLOAdapter(args.model, device=args.device)
    target_ids = resolve_class_ids(adapter.names, args.critical_classes)
    if args.image:
        result = guard_image(adapter, args.image, target_ids)
        print(result)
        if args.out:
            write_json(args.out, result)
    else:
        paths = list_images(args.images, max_images=args.max_images)
        out = args.out or "guard.csv"
        result = guard_batch(adapter, paths, target_ids, out)
        print(result)
        json_out = Path(out).with_suffix(".summary.json")
        write_json(json_out, result)


if __name__ == "__main__":
    main()
