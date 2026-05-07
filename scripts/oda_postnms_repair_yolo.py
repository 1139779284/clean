#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.oda_postnms_repair import ODAPostNMSRepairConfig, run_oda_postnms_repair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-image ODA post-NMS/localized-recall surgical repair for YOLO")
    p.add_argument("--model", required=True, help="Input candidate checkpoint, ideally current Pareto best such as external max ASR 0.15")
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--external-roots", nargs="+", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--attack-names", nargs="*", default=None, help="Usually badnet_oda. If omitted, all discovered ODA attacks are used.")
    p.add_argument("--failure-rows-csv", default=None, help="Optional external_hard_suite_rows.csv. If omitted, failures are discovered from --model.")
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--failure-repeat", type=int, default=24)
    p.add_argument("--clean-anchor-images", type=int, default=8)
    p.add_argument("--max-single-attack-worsen", type=float, default=0.02)
    p.add_argument("--max-allowed-external-asr", type=float, default=0.10)
    p.add_argument("--lambda-task", type=float, default=0.03)
    p.add_argument("--lambda-oda-matched", type=float, default=4.0)
    p.add_argument("--lambda-oga-negative", type=float, default=0.0)
    p.add_argument("--oda-min-score", type=float, default=0.60)
    p.add_argument("--oda-localized-margin", type=float, default=0.20)
    p.add_argument("--oda-localized-margin-weight", type=float, default=0.90)
    p.add_argument("--oda-best-score-weight", type=float, default=1.25)
    p.add_argument("--oda-best-box-weight", type=float, default=0.55)
    p.add_argument("--oda-box-weight", type=float, default=0.45)
    p.add_argument("--no-require-improvement", action="store_true", help="Allow best unblocked candidate as final even if external score did not improve. Not recommended for final reports.")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ODAPostNMSRepairConfig(
        model=args.model,
        data_yaml=args.data_yaml,
        out_dir=args.out,
        external_roots=tuple(args.external_roots),
        target_classes=tuple(args.target_classes),
        attack_names=tuple(args.attack_names or ()),
        failure_rows_csv=args.failure_rows_csv,
        teacher_model=args.teacher_model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_images_per_attack=args.max_images_per_attack,
        replay_max_images_per_attack=args.replay_max_images_per_attack,
        failure_repeat=args.failure_repeat,
        clean_anchor_images=args.clean_anchor_images,
        max_single_attack_worsen=args.max_single_attack_worsen,
        max_allowed_external_asr=args.max_allowed_external_asr,
        lambda_task=args.lambda_task,
        lambda_oda_matched=args.lambda_oda_matched,
        lambda_oga_negative=args.lambda_oga_negative,
        oda_min_score=args.oda_min_score,
        oda_localized_margin=args.oda_localized_margin,
        oda_localized_margin_weight=args.oda_localized_margin_weight,
        oda_best_score_weight=args.oda_best_score_weight,
        oda_best_box_weight=args.oda_best_box_weight,
        oda_box_weight=args.oda_box_weight,
        require_improvement_for_final=not bool(args.no_require_improvement),
        amp=bool(args.amp),
    )
    manifest = run_oda_postnms_repair(cfg)
    print(json.dumps({
        "status": manifest.get("status"),
        "rolled_back": manifest.get("rolled_back"),
        "final_model": manifest.get("final_model"),
        "before": manifest.get("before_summary"),
        "best": manifest.get("best"),
    }, ensure_ascii=False, indent=2))
    print(f"[DONE] manifest: {Path(args.out) / 'oda_postnms_repair_manifest.json'}")


if __name__ == "__main__":
    main()
