#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.io import write_json
from model_security_gate.verify.acceptance_gate import compare_security_reports, compare_yolo_metrics, decide_acceptance


def _load(path: str | None):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decide whether a detoxed model passes acceptance")
    p.add_argument("--before-report", required=True)
    p.add_argument("--after-report", required=True)
    p.add_argument("--before-metrics", default=None)
    p.add_argument("--after-metrics", default=None)
    p.add_argument("--detox-manifest", default=None, help="Optional strong_detox_manifest.json for weak-supervision and verification-status gates")
    p.add_argument("--out", default="runs/acceptance.json")
    p.add_argument("--max-map-drop", type=float, default=0.03)
    p.add_argument("--min-fp-reduction", type=float, default=0.8)
    p.add_argument("--allow-weak-supervision", action="store_true", help="Allow feature_only/self-pseudo runs to pass acceptance. Not recommended for production gates.")
    p.add_argument("--compare-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    before_report = _load(args.before_report)
    after_report = _load(args.after_report)
    before_metrics = _load(args.before_metrics)
    after_metrics = _load(args.after_metrics)
    detox_manifest = _load(args.detox_manifest)
    if args.compare_only:
        result = {
            "security_compare": compare_security_reports(before_report, after_report),
            "metric_compare": compare_yolo_metrics(before_metrics, after_metrics) if before_metrics and after_metrics else {"available": False},
        }
    else:
        result = decide_acceptance(
            before_report=before_report,
            after_report=after_report,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_map_drop=args.max_map_drop,
            min_fp_reduction=args.min_fp_reduction,
            detox_manifest=detox_manifest,
            allow_weak_supervision=args.allow_weak_supervision,
        )
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    main()
