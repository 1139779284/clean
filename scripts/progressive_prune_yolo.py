#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Build and select soft-pruned YOLO candidates using counterfactual TTA risk")
    p.add_argument("--model", required=True)
    p.add_argument("--channel-csv", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--out", default="runs/progressive_prune")
    p.add_argument("--top-ks", type=int, nargs="*", default=[10, 25, 50, 100])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default=None)
    p.add_argument("--max-eval-images", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    from model_security_gate.detox.progressive_prune import ProgressivePruneConfig, progressive_prune_and_select
    from model_security_gate.utils.io import list_images, load_class_names_from_data_yaml, resolve_class_ids

    names = load_class_names_from_data_yaml(args.data_yaml)
    target_ids = resolve_class_ids(names, args.target_classes)
    paths = list_images(args.images, max_images=args.max_eval_images)
    df = pd.read_csv(args.channel_csv)
    manifest = progressive_prune_and_select(
        model_path=args.model,
        ranked_channels=df,
        image_paths=paths,
        labels_dir=args.labels,
        target_class_ids=target_ids,
        output_dir=args.out,
        cfg=ProgressivePruneConfig(top_ks=tuple(args.top_ks), conf=args.conf, iou=args.iou, imgsz=args.imgsz, max_eval_images=args.max_eval_images),
        device=args.device,
    )
    print(f"[DONE] selected: {manifest.get('selected')}")
    print(f"[DONE] manifest: {Path(args.out) / 'progressive_prune_manifest.json'}")


if __name__ == "__main__":
    main()
