#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.targeted_repair import TargetedRepairConfig, run_targeted_repair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic failure-only YOLO repair using external hard-suite failures")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--external-roots", nargs="+", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--attack-names", nargs="*", default=None)
    p.add_argument("--repair-goal", choices=["oda", "oga", "mixed", "all"], default="oda")
    p.add_argument("--failure-rows-csv", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--failure-repeat", type=int, default=12)
    p.add_argument("--oda-full-image-extra-repeat", type=int, default=0)
    p.add_argument("--oda-focus-crops", action="store_true")
    p.add_argument("--oda-focus-crop-repeat", type=int, default=2)
    p.add_argument("--clean-anchor-images", type=int, default=24)
    p.add_argument("--max-single-attack-worsen", type=float, default=0.02)
    p.add_argument("--max-allowed-external-asr", type=float, default=0.10)
    p.add_argument("--lambda-task", type=float, default=0.30)
    p.add_argument("--lambda-oda-recall", type=float, default=0.80)
    p.add_argument("--lambda-oda-matched", type=float, default=2.00)
    p.add_argument("--lambda-oga-negative", type=float, default=0.00)
    p.add_argument("--lambda-attention", type=float, default=0.00)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TargetedRepairConfig(
        model=args.model,
        data_yaml=args.data_yaml,
        out_dir=args.out,
        external_roots=tuple(args.external_roots),
        target_classes=tuple(args.target_classes),
        attack_names=tuple(args.attack_names or ()),
        repair_goal=args.repair_goal,
        failure_rows_csv=args.failure_rows_csv,
        teacher_model=args.teacher_model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_images_per_attack=args.max_images_per_attack,
        replay_max_images_per_attack=args.replay_max_images_per_attack,
        failure_repeat=args.failure_repeat,
        oda_full_image_extra_repeat=args.oda_full_image_extra_repeat,
        oda_focus_crops=bool(args.oda_focus_crops),
        oda_focus_crop_repeat=args.oda_focus_crop_repeat,
        clean_anchor_images=args.clean_anchor_images,
        max_single_attack_worsen=args.max_single_attack_worsen,
        max_allowed_external_asr=args.max_allowed_external_asr,
        lambda_task=args.lambda_task,
        lambda_oda_recall=args.lambda_oda_recall,
        lambda_oda_matched=args.lambda_oda_matched,
        lambda_oga_negative=args.lambda_oga_negative,
        lambda_attention=args.lambda_attention,
        weight_decay=args.weight_decay,
    )
    manifest = run_targeted_repair(cfg)
    print(json.dumps({
        "status": manifest.get("status"),
        "final_model": manifest.get("final_model"),
        "before": manifest.get("before_summary"),
        "best": manifest.get("best"),
    }, ensure_ascii=False, indent=2))
    print(f"[DONE] manifest: {Path(args.out) / 'targeted_repair_manifest.json'}")


if __name__ == "__main__":
    main()

