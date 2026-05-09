#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.evidence_gate import T0EvidenceGateConfig
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
    p.add_argument("--max-guard-free-asr", type=float, default=0.05)
    p.add_argument("--max-guard-free-mean-asr", type=float, default=0.02)
    p.add_argument("--max-trigger-only-asr", type=float, default=0.05)
    p.add_argument("--max-map-drop", type=float, default=0.03)
    p.add_argument("--confidence", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = T0EvidenceGateConfig(
        max_guard_free_asr=float(args.max_guard_free_asr),
        max_guard_free_mean_asr=float(args.max_guard_free_mean_asr),
        max_trigger_only_asr=float(args.max_trigger_only_asr),
        max_clean_map50_95_drop=float(args.max_map_drop),
        confidence=float(args.confidence),
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
        cfg=cfg,
    )
    print(f"[DONE] tier={payload['gate']['tier']} accepted={payload['gate']['accepted']}")
    print(f"[DONE] report: {Path(args.out) / 'T0_EVIDENCE_PACK.md'}")


if __name__ == "__main__":
    main()
