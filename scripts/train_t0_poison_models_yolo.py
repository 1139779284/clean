#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.attack_zoo.poison_train import PoisonDatasetConfig, build_poison_train_dataset
from model_security_gate.attack_zoo.specs import load_attack_specs


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build poisoned datasets and train YOLO poison-model matrix entries")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out", default="runs/t0_poison_models")
    parser.add_argument("--attack-config", default="configs/t0_attack_zoo.yaml")
    parser.add_argument("--attacks", default="badnet_oga_corner,semantic_cleanlabel,wanet_oga")
    parser.add_argument("--poison-rates", default="0.05")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--source-class-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = {spec.name: spec for spec in load_attack_specs(args.attack_config)}
    attacks = _split_csv(args.attacks)
    poison_rates = [float(item) for item in _split_csv(args.poison_rates)]
    seeds = [int(item) for item in _split_csv(args.seeds)]
    missing = [attack for attack in attacks if attack not in specs]
    if missing:
        raise SystemExit(f"unknown attacks: {missing}; available={sorted(specs)}")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for attack_name in attacks:
        for poison_rate in poison_rates:
            for seed in seeds:
                run_name = f"{attack_name}_pr{int(round(poison_rate * 10000)):04d}_seed{seed}"
                dataset_root = out_root / "datasets" / run_name
                train_project = (out_root / "training").resolve()
                train_dir = train_project / run_name
                weights_path = train_dir / "weights" / "best.pt"
                dataset_result = build_poison_train_dataset(
                    PoisonDatasetConfig(
                        clean_root=args.clean_root,
                        out_root=str(dataset_root.resolve()),
                        attack=specs[attack_name],
                        poison_rate=poison_rate,
                        seed=seed,
                        target_class_id=args.target_class_id,
                        source_class_id=args.source_class_id,
                        max_train_images=args.max_train_images,
                    )
                )
                row: dict[str, object] = {
                    "attack": attack_name,
                    "poison_rate": poison_rate,
                    "seed": seed,
                    "dataset": dataset_result.to_dict(),
                    "training_dir": str(train_dir),
                    "weights": str(weights_path),
                    "status": "planned",
                }
                if args.dry_run:
                    rows.append(row)
                    continue
                if args.skip_existing and weights_path.exists():
                    row["status"] = "skipped_existing"
                    rows.append(row)
                    continue
                from ultralytics import YOLO

                model = YOLO(args.base_model)
                model.train(
                    data=dataset_result.data_yaml,
                    epochs=int(args.epochs),
                    imgsz=int(args.imgsz),
                    batch=int(args.batch),
                    workers=int(args.workers),
                    device=str(args.device),
                    project=str(train_project),
                    name=run_name,
                    exist_ok=True,
                )
                if not weights_path.exists():
                    matches = sorted(train_project.rglob(f"{run_name}/weights/best.pt"))
                    if matches:
                        weights_path = matches[-1]
                        row["weights"] = str(weights_path)
                        row["training_dir"] = str(weights_path.parents[1])
                row["status"] = "trained" if weights_path.exists() else "trained_missing_best"
                rows.append(row)
                (out_root / "poison_training_manifest.json").write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "clean_root": args.clean_root,
        "base_model": args.base_model,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "dry_run": bool(args.dry_run),
        "runs": rows,
    }
    (out_root / "poison_training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'poison_training_manifest.json'} runs={len(rows)} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
