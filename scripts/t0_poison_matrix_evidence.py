#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.poison_matrix_evidence import (
    PoisonMatrixEvidenceConfig,
    build_poison_matrix_evidence,
    write_poison_matrix_evidence,
)


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _expand_reports(patterns: list[str]) -> list[str]:
    reports: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        reports.extend(matches or [pattern])
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build T0 poison-model matrix evidence from trained poison model ASR reports")
    parser.add_argument("--config", default=None, help="Optional YAML config with expected_attacks and ASR thresholds")
    parser.add_argument("--summary-json", default=None, help="Matrix summary JSON containing entries with weights/report fields")
    parser.add_argument("--reports", nargs="*", default=[], help="External hard-suite report paths or glob patterns")
    parser.add_argument("--root", default=".", help="Root used to resolve relative weight/report paths")
    parser.add_argument("--out", default="runs/t0_poison_matrix_evidence")
    parser.add_argument("--expected-attacks", default="badnet_oga_corner,semantic_cleanlabel,wanet_oga")
    parser.add_argument("--expected-seeds", default="", help="Optional comma-separated seeds required for full-factorial coverage")
    parser.add_argument("--expected-poison-rates", default="", help="Optional comma-separated poison rates required for full-factorial coverage")
    parser.add_argument("--min-primary-asr", type=float, default=0.20)
    parser.add_argument("--min-usable-asr", type=float, default=0.05)
    parser.add_argument("--require-full-factorial", action="store_true")
    parser.add_argument("--full-factorial-cell-acceptance", choices=["present", "usable", "strong"], default="strong")
    parser.add_argument("--no-require-any-strong", action="store_true")
    parser.add_argument("--allow-missing-weights", action="store_true")
    parser.add_argument("--allow-missing-reports", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    expected_attacks = config.get("expected_attacks")
    if isinstance(expected_attacks, list):
        expected = tuple(str(item) for item in expected_attacks)
    else:
        expected = _split_csv(args.expected_attacks)
    expected_seeds = config.get("expected_seeds")
    if isinstance(expected_seeds, list):
        seeds = tuple(int(item) for item in expected_seeds)
    else:
        seeds = tuple(int(item) for item in _split_csv(args.expected_seeds))
    expected_poison_rates = config.get("expected_poison_rates")
    if isinstance(expected_poison_rates, list):
        poison_rates = tuple(float(item) for item in expected_poison_rates)
    else:
        poison_rates = tuple(float(item) for item in _split_csv(args.expected_poison_rates))
    cfg = PoisonMatrixEvidenceConfig(
        expected_attacks=expected,
        expected_seeds=seeds,
        expected_poison_rates=poison_rates,
        min_primary_asr=float(config.get("min_primary_asr", args.min_primary_asr)),
        min_usable_asr=float(config.get("min_usable_asr", args.min_usable_asr)),
        require_weights=bool(config.get("require_weights", not bool(args.allow_missing_weights))),
        require_report=bool(config.get("require_report", not bool(args.allow_missing_reports))),
        require_any_strong=bool(config.get("require_any_strong", not bool(args.no_require_any_strong))),
        require_full_factorial=bool(config.get("require_full_factorial", bool(args.require_full_factorial))),
        full_factorial_cell_acceptance=str(config.get("full_factorial_cell_acceptance", args.full_factorial_cell_acceptance)),
    )
    evidence = build_poison_matrix_evidence(
        summary_json=args.summary_json,
        report_paths=_expand_reports(list(args.reports)),
        root=args.root,
        cfg=cfg,
    )
    json_path, md_path = write_poison_matrix_evidence(args.out, evidence)
    print(f"[DONE] status={evidence['status']} accepted={evidence['accepted']}")
    print(f"[DONE] json: {json_path}")
    print(f"[DONE] report: {md_path}")
    return 0 if evidence["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
