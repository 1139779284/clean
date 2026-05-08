#!/usr/bin/env python3
"""Strict production Green gate CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_security_gate.verify.green_gate import evaluate_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detox candidate against production Green constraints")
    parser.add_argument("--after-report", default=None, help="security_report.json after repair")
    parser.add_argument("--before-metrics", default=None, help="Clean metrics JSON before repair")
    parser.add_argument("--after-metrics", default=None, help="Clean metrics JSON after repair")
    parser.add_argument("--external-result", required=True, help="external_hard_suite result JSON after repair")
    parser.add_argument("--baseline-external-result", default=None, help="Baseline external_hard_suite result JSON")
    parser.add_argument("--weak-supervision-report", default=None, help="Pseudo-label/teacher quality report JSON")
    parser.add_argument("--config", default=None, help="production_green_gate.yaml")
    parser.add_argument("--output", default="production_green_gate.json", help="Output decision JSON")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when blocked")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_from_files(
        after_report=args.after_report,
        before_metrics=args.before_metrics,
        after_metrics=args.after_metrics,
        external_result=args.external_result,
        baseline_external_result=args.baseline_external_result,
        weak_supervision_report=args.weak_supervision_report,
        config_path=args.config,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.accepted or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
