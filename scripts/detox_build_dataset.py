#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.dataset_builder import DetoxDatasetConfig, build_counterfactual_yolo_dataset
from model_security_gate.utils.io import load_class_names_from_data_yaml, resolve_class_ids


def parse_args():
    p = argparse.ArgumentParser(description="Build counterfactual YOLO dataset for trigger-agnostic detox fine-tuning")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--data-yaml", required=True, help="Original data.yaml containing class names")
    p.add_argument("--out", default="runs/detox_dataset")
    p.add_argument("--target-classes", nargs="+", required=True, help="Class names or IDs to remove in target-removal counterfactuals")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="*", default=None, help="Override default variants")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = load_class_names_from_data_yaml(args.data_yaml)
    target_ids = resolve_class_ids(names, args.target_classes)
    cfg = DetoxDatasetConfig(val_fraction=args.val_fraction, seed=args.seed, variants=args.variants)
    data_yaml = build_counterfactual_yolo_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.out,
        class_names=names,
        target_class_ids=target_ids,
        cfg=cfg,
    )
    print(f"[DONE] data yaml: {data_yaml}")
    print(f"[INFO] target IDs: {target_ids}")


if __name__ == "__main__":
    main()
