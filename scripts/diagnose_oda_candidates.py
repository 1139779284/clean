#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.oda_candidate_diagnostics import ODACandidateDiagnosticConfig, diagnose_oda_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose whether ODA failures are pre-NMS candidate, score, or final post-NMS issues")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--external-roots", nargs="*", default=None, help="Benchmark roots. Used when --rows-csv is not provided.")
    p.add_argument("--attack-names", nargs="*", default=None, help="Usually badnet_oda. If omitted, all ODA failures are diagnosed.")
    p.add_argument("--rows-csv", default=None, help="Optional existing external_hard_suite_rows.csv.")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--low-conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--match-iou", type=float, default=0.30)
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--raw-topk", type=int, default=64)
    p.add_argument("--raw-center-radius", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ODACandidateDiagnosticConfig(
        model=args.model,
        data_yaml=args.data_yaml,
        out_dir=args.out,
        target_classes=tuple(args.target_classes),
        external_roots=tuple(args.external_roots or ()),
        attack_names=tuple(args.attack_names or ()),
        rows_csv=args.rows_csv,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        low_conf=args.low_conf,
        iou=args.iou,
        match_iou=args.match_iou,
        max_images_per_attack=args.max_images_per_attack,
        raw_topk=args.raw_topk,
        raw_center_radius=args.raw_center_radius,
    )
    if not cfg.rows_csv and not cfg.external_roots:
        raise SystemExit("Either --rows-csv or --external-roots is required.")
    result = diagnose_oda_candidates(cfg)
    print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False))
    print(f"[DONE] report: {Path(args.out) / 'oda_candidate_diagnostics.json'}")


if __name__ == "__main__":
    main()
