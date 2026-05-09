#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.benchmark_audit import BenchmarkAuditConfig, audit_benchmark, write_audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit external hard-suite integrity for T0-style claims")
    p.add_argument("--roots", nargs="+", required=True, help="Benchmark roots to audit")
    p.add_argument("--target-class-id", type=int, default=0)
    p.add_argument("--suppressor-class-id", type=int, default=1)
    p.add_argument("--heldout-roots", nargs="*", default=[])
    p.add_argument("--min-images-per-attack", type=int, default=0)
    p.add_argument("--out", default="runs/t0_benchmark_audit/t0_benchmark_audit.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkAuditConfig(
        roots=tuple(args.roots),
        target_class_id=int(args.target_class_id),
        suppressor_class_id=int(args.suppressor_class_id),
        min_images_per_attack=int(args.min_images_per_attack),
        heldout_roots=tuple(args.heldout_roots or ()),
    )
    result = audit_benchmark(cfg)
    write_audit(args.out, result)
    print(f"[DONE] passed={result['passed']} records={result['n_records']} attacks={result['n_attacks']} errors={result['n_errors']} warnings={result['n_warnings']}")
    print(f"[DONE] report: {args.out}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
