#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.evidence_gate import T0EvidenceGateConfig
from model_security_gate.t0.matrix_aggregator import MatrixAggregatorConfig
from model_security_gate.t0.report import build_t0_evidence_pack


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a T0 evidence pack from existing validation outputs")
    p.add_argument("--out", default="runs/t0_evidence_pack")
    p.add_argument("--guard-free-external", default=None, help="Corrected, guard-free external_hard_suite_asr.json")
    p.add_argument("--guarded-external", default=None, help="Guarded external_hard_suite_asr.json for deployment layer")
    p.add_argument("--trigger-only-external", default=None, help="Trigger-only filtered external_hard_suite_asr.json")
    p.add_argument("--clean-before", default=None, help="Clean metrics before detox")
    p.add_argument("--clean-after", default=None, help="Clean metrics after detox")
    p.add_argument("--benchmark-audit", default=None, help="t0_benchmark_audit.json")
    p.add_argument("--heldout-leakage", default=None, help="heldout leakage manifest")
    p.add_argument(
        "--poison-matrix-summary",
        action="append",
        default=[],
        help=(
            "Poison-matrix summary JSON (repeatable). Entries from all summaries are "
            "deduplicated and aggregated into per-attack pass rates, dose-response "
            "curves, and off-target bleed-over."
        ),
    )
    p.add_argument("--matrix-strong-asr-threshold", type=float, default=0.20)
    p.add_argument("--matrix-usable-asr-threshold", type=float, default=0.05)
    p.add_argument("--matrix-off-target-warn-absolute", type=float, default=0.20)
    p.add_argument("--matrix-off-target-warn-delta", type=float, default=0.15)
    p.add_argument("--matrix-dose-response-tolerance", type=float, default=0.02)
    p.add_argument(
        "--skip-full-matrix-report",
        action="store_true",
        help="Embed the aggregate inside the evidence pack only; do not write the stand-alone markdown.",
    )
    p.add_argument(
        "--fail-on-matrix-warnings",
        action="store_true",
        help="Exit non-zero if the poison matrix aggregate has warnings.",
    )
    p.add_argument("--max-guard-free-asr", type=float, default=0.05)
    p.add_argument("--max-guard-free-mean-asr", type=float, default=0.02)
    p.add_argument("--max-trigger-only-asr", type=float, default=0.05)
    p.add_argument("--max-map-drop", type=float, default=0.03)
    p.add_argument("--confidence", type=float, default=0.95)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = T0EvidenceGateConfig(
        max_guard_free_asr=float(args.max_guard_free_asr),
        max_guard_free_mean_asr=float(args.max_guard_free_mean_asr),
        max_trigger_only_asr=float(args.max_trigger_only_asr),
        max_clean_map50_95_drop=float(args.max_map_drop),
        confidence=float(args.confidence),
    )
    matrix_cfg = MatrixAggregatorConfig(
        strong_asr_threshold=float(args.matrix_strong_asr_threshold),
        usable_asr_threshold=float(args.matrix_usable_asr_threshold),
        confidence=float(args.confidence),
        off_target_warn_absolute=float(args.matrix_off_target_warn_absolute),
        off_target_warn_delta=float(args.matrix_off_target_warn_delta),
        dose_response_tolerance=float(args.matrix_dose_response_tolerance),
    )
    payload = build_t0_evidence_pack(
        out_dir=args.out,
        guard_free_external=args.guard_free_external,
        guarded_external=args.guarded_external,
        trigger_only_external=args.trigger_only_external,
        clean_metrics_before=args.clean_before,
        clean_metrics_after=args.clean_after,
        benchmark_audit=args.benchmark_audit,
        heldout_leakage=args.heldout_leakage,
        poison_matrix_summaries=list(args.poison_matrix_summary) or None,
        matrix_config=matrix_cfg,
        write_full_matrix_report=not bool(args.skip_full_matrix_report),
        cfg=cfg,
    )
    gate = payload.get("gate") or {}
    print(f"[DONE] tier={gate.get('tier')} accepted={gate.get('accepted')}")
    print(f"[DONE] report: {Path(args.out) / 'T0_EVIDENCE_PACK.md'}")
    matrix = payload.get("matrix_aggregate") or {}
    if matrix:
        overall = matrix.get("overall") or {}
        strong = overall.get("strong_cell_pass_rate") or {}
        print(
            "[DONE] matrix status={status} entries={n} strong={s}/{t}".format(
                status=matrix.get("status"),
                n=matrix.get("n_entries"),
                s=strong.get("successes", 0),
                t=strong.get("total", 0),
            )
        )
    if args.fail_on_matrix_warnings and matrix and matrix.get("warnings"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
