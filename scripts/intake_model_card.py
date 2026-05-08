#!/usr/bin/env python3
"""Run formal model intake and write an intake manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from model_security_gate.intake.formal_intake import load_intake_config, run_formal_intake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal intake checks for a YOLO model artifact")
    parser.add_argument("--model", required=True, help="Path to model artifact, e.g. .pt")
    parser.add_argument("--model-card", default=None, help="YAML/JSON model card")
    parser.add_argument("--training-log", default=None, help="Training log or experiment summary")
    parser.add_argument("--data-yaml", default=None, help="YOLO data.yaml containing class names")
    parser.add_argument("--preprocess", default=None, help="YAML/JSON preprocess contract")
    parser.add_argument("--provenance", default=None, help="YAML/JSON provenance contract")
    parser.add_argument("--config", default=None, help="Formal intake YAML/JSON config")
    parser.add_argument("--output", default="intake_manifest.json", help="Output manifest path")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when intake is blocked")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_intake_config(args.config)
    result = run_formal_intake(
        model_path=args.model,
        output_path=args.output,
        model_card_path=args.model_card,
        training_log_path=args.training_log,
        data_yaml_path=args.data_yaml,
        preprocess_path=args.preprocess,
        provenance_path=args.provenance,
        config=cfg,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if result.accepted or args.allow_blocked:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
