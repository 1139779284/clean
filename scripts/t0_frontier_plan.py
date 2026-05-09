#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.metrics import load_json
from model_security_gate.t0.residuals import build_frontier_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan next frontier detox phases from residual ASR reports")
    p.add_argument("--reports", nargs="+", required=True, help="external_hard_suite_asr.json files")
    p.add_argument("--min-asr", type=float, default=0.001)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out", default="runs/t0_frontier_plan/t0_frontier_plan.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = [load_json(p) for p in args.reports]
    plan = build_frontier_plan(reports, min_asr=float(args.min_asr), top_k=int(args.top_k))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] recommended phases:")
    for phase in plan.get("recommended_phase_order", []):
        print(f"  - {phase}")
    print(f"[DONE] report: {out}")


if __name__ == "__main__":
    main()
