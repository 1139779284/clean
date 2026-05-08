#!/usr/bin/env python3
"""Evaluate numeric candidate metrics against joint no-worse constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _load_structured(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint no-worse scorecard for candidate detox metrics")
    parser.add_argument("--metrics", required=True, help="JSON/YAML metrics map")
    parser.add_argument("--config", default=None, help="joint_no_worse_repair.yaml")
    parser.add_argument("--output", default="joint_no_worse_scorecard.json")
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from model_security_gate.detox.joint_no_worse import (  # lazy import keeps --help lightweight
        AttackNoWorseSpec,
        JointNoWorseConfig,
        candidate_no_worse_scorecard,
        merge_scorecard_metrics,
    )

    metrics_data = _load_structured(args.metrics)
    metrics = merge_scorecard_metrics(metrics_data.get("metrics", metrics_data))

    if args.config:
        cfg_data = _load_structured(args.config)
        nw_data = cfg_data.get("joint_no_worse", cfg_data)
        specs_data = nw_data.get("specs") if isinstance(nw_data, dict) else None
        if specs_data:
            specs = [AttackNoWorseSpec(**dict(item)) for item in specs_data]
        else:
            specs = JointNoWorseConfig.production_defaults().specs
    else:
        specs = JointNoWorseConfig.production_defaults().specs

    scorecard = candidate_no_worse_scorecard(metrics, specs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False))
    return 0 if scorecard.accepted or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
