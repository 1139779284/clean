#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, write_resolved_config


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on a counterfactual detox dataset")
    p.add_argument("--config", default=None, help="YAML config. Values under `detox_train:` are also accepted. CLI args override YAML.")
    p.add_argument("--base-model", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--out-project", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = {
        "base_model": None,
        "data_yaml": None,
        "out_project": "runs/detox_train",
        "name": "detox_yolo",
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "device": None,
    }
    cfg = load_yaml_config(args.config, section="detox_train")
    resolved = deep_merge(defaults, deep_merge(cfg, namespace_overrides(args, exclude={"config"})))
    if not resolved.get("base_model") or not resolved.get("data_yaml"):
        raise SystemExit("--base-model and --data-yaml are required, or provide them in --config")
    out_project = Path(str(resolved.pop("out_project")))
    write_resolved_config(out_project / str(resolved.get("name", "detox_yolo")) / "resolved_config.json", {"out_project": str(out_project), **resolved})
    base_model = resolved.pop("base_model")
    data_yaml = resolved.pop("data_yaml")
    name = resolved.pop("name")
    train_counterfactual_finetune(
        base_model=base_model,
        data_yaml=data_yaml,
        output_project=out_project,
        name=name,
        **resolved,
    )


if __name__ == "__main__":
    main()
