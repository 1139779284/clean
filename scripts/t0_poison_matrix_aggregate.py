#!/usr/bin/env python3
"""CLI for T0 poison-matrix aggregate evidence.

Reads one or more poison-matrix summary JSON files (the shape produced by
``scripts/train_t0_poison_models_yolo.py`` and stored in
``runs/t0_poison_model_matrix_summary_*``) and writes a publication-grade
aggregate report with per-attack Wilson pass rates, dose-response curves,
seed stability, and off-target bleed-over matrices.

Example:

    pixi run python scripts/t0_poison_matrix_aggregate.py \
        --summary-json runs/t0_poison_model_matrix_summary_2026-05-10/t0_poison_model_matrix_summary.json \
        --out runs/t0_poison_matrix_aggregate_2026-05-10

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.matrix_aggregator import (
    MatrixAggregatorConfig,
    aggregate_matrix_entries,
    write_matrix_aggregate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate poison-matrix evidence across attacks, seeds, and poison rates")
    parser.add_argument(
        "--summary-json",
        action="append",
        default=[],
        help="Path to poison-matrix summary JSON (repeatable to merge multiple summaries)",
    )
    parser.add_argument("--out", default="runs/t0_poison_matrix_aggregate")
    parser.add_argument("--strong-asr-threshold", type=float, default=0.20)
    parser.add_argument("--usable-asr-threshold", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--off-target-warn-absolute", type=float, default=0.20)
    parser.add_argument("--off-target-warn-delta", type=float, default=0.15)
    parser.add_argument("--dose-response-tolerance", type=float, default=0.02)
    parser.add_argument("--min-seeds-for-stability", type=int, default=2)
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser.parse_args()


def _merge_entries(paths: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("run") or entry.get("weights") or json.dumps(entry, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


def main() -> int:
    args = parse_args()
    if not args.summary_json:
        print("[ERROR] at least one --summary-json is required", file=sys.stderr)
        return 2
    entries = _merge_entries(args.summary_json)
    if not entries:
        print("[ERROR] no entries found in summary JSON(s)", file=sys.stderr)
        return 2
    cfg = MatrixAggregatorConfig(
        strong_asr_threshold=float(args.strong_asr_threshold),
        usable_asr_threshold=float(args.usable_asr_threshold),
        confidence=float(args.confidence),
        off_target_warn_absolute=float(args.off_target_warn_absolute),
        off_target_warn_delta=float(args.off_target_warn_delta),
        dose_response_tolerance=float(args.dose_response_tolerance),
        min_seeds_for_stability=int(args.min_seeds_for_stability),
    )
    aggregate = aggregate_matrix_entries(entries, cfg=cfg)
    json_path, md_path = write_matrix_aggregate(args.out, aggregate)
    print(f"[DONE] status={aggregate['status']} entries={aggregate['n_entries']}")
    print(f"[DONE] json: {json_path}")
    print(f"[DONE] report: {md_path}")
    if args.fail_on_warnings and aggregate.get("warnings"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
