#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.oda_score_calibration_repair import (
    ODAScoreCalibrationRepairConfig,
    run_oda_score_calibration_repair,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Near-GT target-score calibration repair for residual ODA failures")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--external-roots", nargs="+", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--attack-names", nargs="*", default=None, help="Usually badnet_oda")
    p.add_argument("--failure-rows-csv", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--low-conf", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--letterbox-train", action="store_true", help="Use letterbox preprocessing in the repair dataloader to match Ultralytics predict.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--failure-repeat", type=int, default=32)
    p.add_argument("--clean-anchor-images", type=int, default=0)
    p.add_argument("--guard-attack-names", nargs="*", default=None, help="Target-absent OGA attacks used as negative guards; defaults to discovered OGA attacks.")
    p.add_argument("--guard-replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--guard-repeat", type=int, default=8)
    p.add_argument("--guard-failure-only", action="store_true", help="Replay only current success=true rows for guard attacks.")
    p.add_argument("--lambda-score-calibration", type=float, default=8.0)
    p.add_argument("--lambda-task", type=float, default=0.0)
    p.add_argument("--lambda-oga-negative", type=float, default=0.0)
    p.add_argument("--lambda-semantic-negative", type=float, default=0.0)
    p.add_argument("--lambda-semantic-fp-region", type=float, default=0.0)
    p.add_argument("--score-conf-target", type=float, default=0.35)
    p.add_argument("--score-margin", type=float, default=0.15)
    p.add_argument("--score-topk-near", type=int, default=24)
    p.add_argument("--score-topk-far", type=int, default=128)
    p.add_argument("--score-positive-bce-weight", type=float, default=0.45)
    p.add_argument("--score-floor-weight", type=float, default=1.0)
    p.add_argument("--score-far-margin-weight", type=float, default=0.55)
    p.add_argument("--score-competing-margin-weight", type=float, default=0.35)
    p.add_argument("--score-teacher-weight", type=float, default=0.35)
    p.add_argument("--semantic-guard-keywords", nargs="*", default=None)
    p.add_argument("--semantic-negative-topk", type=int, default=256)
    p.add_argument("--semantic-negative-max-score", type=float, default=0.05)
    p.add_argument("--semantic-negative-margin-weight", type=float, default=0.50)
    p.add_argument("--semantic-fp-region-topk", type=int, default=64)
    p.add_argument("--semantic-fp-region-iou-threshold", type=float, default=0.03)
    p.add_argument("--semantic-fp-region-center-radius", type=float, default=2.0)
    p.add_argument("--semantic-fp-region-max-score", type=float, default=0.03)
    p.add_argument("--semantic-fp-region-margin-weight", type=float, default=1.0)
    p.add_argument("--max-single-attack-worsen", type=float, default=0.02)
    p.add_argument("--max-allowed-external-asr", type=float, default=0.10)
    p.add_argument("--min-diagnostic-improvement", type=float, default=0.03)
    p.add_argument("--no-require-external-improvement", action="store_true")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ODAScoreCalibrationRepairConfig(
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
        low_conf=args.low_conf,
        batch=args.batch,
        letterbox_train=args.letterbox_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_images_per_attack=args.max_images_per_attack,
        replay_max_images_per_attack=args.replay_max_images_per_attack,
        failure_repeat=args.failure_repeat,
        clean_anchor_images=args.clean_anchor_images,
        guard_attack_names=tuple(args.guard_attack_names or ()),
        guard_replay_max_images_per_attack=args.guard_replay_max_images_per_attack,
        guard_repeat=args.guard_repeat,
        guard_failure_only=args.guard_failure_only,
        lambda_score_calibration=args.lambda_score_calibration,
        lambda_task=args.lambda_task,
        lambda_oga_negative=args.lambda_oga_negative,
        lambda_semantic_negative=args.lambda_semantic_negative,
        lambda_semantic_fp_region=args.lambda_semantic_fp_region,
        score_conf_target=args.score_conf_target,
        score_margin=args.score_margin,
        score_topk_near=args.score_topk_near,
        score_topk_far=args.score_topk_far,
        score_positive_bce_weight=args.score_positive_bce_weight,
        score_floor_weight=args.score_floor_weight,
        score_far_margin_weight=args.score_far_margin_weight,
        score_competing_margin_weight=args.score_competing_margin_weight,
        score_teacher_weight=args.score_teacher_weight,
        semantic_guard_keywords=tuple(args.semantic_guard_keywords or ("semantic",)),
        semantic_negative_topk=args.semantic_negative_topk,
        semantic_negative_max_score=args.semantic_negative_max_score,
        semantic_negative_margin_weight=args.semantic_negative_margin_weight,
        semantic_fp_region_topk=args.semantic_fp_region_topk,
        semantic_fp_region_iou_threshold=args.semantic_fp_region_iou_threshold,
        semantic_fp_region_center_radius=args.semantic_fp_region_center_radius,
        semantic_fp_region_max_score=args.semantic_fp_region_max_score,
        semantic_fp_region_margin_weight=args.semantic_fp_region_margin_weight,
        max_single_attack_worsen=args.max_single_attack_worsen,
        max_allowed_external_asr=args.max_allowed_external_asr,
        min_diag_score_improvement=args.min_diagnostic_improvement,
        require_external_improvement_for_final=not args.no_require_external_improvement,
        amp=args.amp,
    )
    manifest = run_oda_score_calibration_repair(cfg)
    print(json.dumps({k: manifest.get(k) for k in ("status", "rolled_back", "final_model", "best", "best_by_diagnostic")}, indent=2, ensure_ascii=False))
    print(f"[DONE] manifest: {Path(args.out) / 'oda_score_calibration_repair_manifest.json'}")


if __name__ == "__main__":
    main()
