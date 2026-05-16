#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.attack_zoo.poison_train import PoisonDatasetConfig, build_poison_train_dataset
from model_security_gate.attack_zoo.specs import load_attack_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO train/val dataset with a selected T0 poison attack")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attack-name", required=True)
    parser.add_argument("--attack-config", default="configs/t0_attack_zoo.yaml")
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--source-class-id", type=int, default=1)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--copy-clean", action="store_true", help="Copy clean files instead of hardlinking them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = {spec.name: spec for spec in load_attack_specs(args.attack_config)}
    if args.attack_name not in specs:
        raise SystemExit(f"unknown attack {args.attack_name!r}; available={sorted(specs)}")
    result = build_poison_train_dataset(
        PoisonDatasetConfig(
            clean_root=args.clean_root,
            out_root=args.out,
            attack=specs[args.attack_name],
            poison_rate=args.poison_rate,
            seed=args.seed,
            target_class_id=args.target_class_id,
            source_class_id=args.source_class_id,
            hardlink_clean=not args.copy_clean,
            max_train_images=args.max_train_images,
        )
    )
    print(
        f"built poison dataset attack={result.attack_name} poisoned={result.poisoned_images}/{result.train_images} "
        f"policy={result.label_policy} data={result.data_yaml}"
    )


if __name__ == "__main__":
    main()
