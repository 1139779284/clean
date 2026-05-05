#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.asr_aware_dataset import load_attacks_from_config
from model_security_gate.detox.asr_aware_train import ASRAwareTrainConfig, run_asr_aware_detox_yolo
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run supervised ASR-aware strong detox for YOLO")
    p.add_argument("--config", default=None, help="YAML config. Values under `asr_aware_detox:` are also accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None, help="Backdoored/suspicious YOLO model")
    p.add_argument("--images", default=None, help="Audited image directory")
    p.add_argument("--labels", default=None, help="Audited YOLO labels directory")
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--epochs-per-cycle", type=int, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-allowed-asr", type=float, default=None)
    p.add_argument("--max-map-drop", type=float, default=None)
    p.add_argument("--include-clean-repeat", type=int, default=None)
    p.add_argument("--include-attack-repeat", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--eval-max-images", type=int, default=None)
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> dict:
    defaults = {
        "model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "out": "runs/asr_aware_detox",
        **{k: v for k, v in ASRAwareTrainConfig().__dict__.items() if k != "attack_specs"},
        "attacks": None,
    }
    cfg = load_yaml_config(args.config, section="asr_aware_detox")
    cli = namespace_overrides(args, exclude={"config"})
    if "data_yaml" in cli:
        cli["data_yaml"] = cli.pop("data_yaml")
    return deep_merge(defaults, deep_merge(cfg, cli))


def main() -> None:
    resolved = resolve_args(parse_args())
    missing = [k for k in ["model", "images", "labels", "data_yaml", "target_classes"] if not resolved.get(k)]
    if missing:
        raise SystemExit(f"Missing required config/CLI values: {', '.join(missing)}")
    out = Path(str(resolved["out"]))
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(out / "resolved_config.json", resolved)
    cfg = ASRAwareTrainConfig(
        imgsz=int(resolved["imgsz"]),
        batch=int(resolved["batch"]),
        device=resolved.get("device"),
        seed=int(resolved["seed"]),
        cycles=int(resolved["cycles"]),
        epochs_per_cycle=int(resolved["epochs_per_cycle"]),
        lr0=float(resolved["lr0"]),
        weight_decay=float(resolved["weight_decay"]),
        max_allowed_asr=float(resolved["max_allowed_asr"]),
        max_map_drop=float(resolved["max_map_drop"]),
        val_fraction=float(resolved["val_fraction"]),
        include_clean_repeat=int(resolved["include_clean_repeat"]),
        include_attack_repeat=int(resolved["include_attack_repeat"]),
        max_images=int(resolved.get("max_images") or 0),
        eval_max_images=int(resolved.get("eval_max_images") or 0),
        attack_specs=load_attacks_from_config(resolved.get("attacks")),
    )
    manifest = run_asr_aware_detox_yolo(
        model_path=resolved["model"],
        images_dir=resolved["images"],
        labels_dir=resolved["labels"],
        data_yaml=resolved["data_yaml"],
        target_classes=resolved["target_classes"],
        output_dir=out,
        cfg=cfg,
    )
    print(f"[DONE] status={manifest.get('status')} final_model={manifest.get('final_model')}")
    print(f"[DONE] manifest={out / 'asr_aware_detox_manifest.json'}")


if __name__ == "__main__":
    main()
