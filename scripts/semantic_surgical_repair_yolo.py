#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.semantic_surgical import SemanticSurgicalRepairConfig, run_semantic_surgical_repair


def _parse_name_value(values: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty name in {raw!r}")
        out[name] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frontier last-mile semantic surgical repair for YOLO backdoor detox")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--external-roots", nargs="+", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--semantic-attack-names", nargs="*", default=None)
    p.add_argument("--guard-attack-names", nargs="*", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--letterbox-train", action="store_true")
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--semantic-failure-repeat", type=int, default=24)
    p.add_argument("--guard-repeat", type=int, default=2)
    p.add_argument("--clean-anchor-images", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--candidate-every-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--trainable-scope", default="head_bias", choices=["head_bias", "bias", "head", "detect_head", "last_module", "last_n_parameters", "all"])
    p.add_argument("--trainable-last-n-modules", type=int, default=1)
    p.add_argument("--trainable-last-n-parameters", type=int, default=40)
    p.add_argument("--lambda-semantic-fp-threshold", type=float, default=8.0)
    p.add_argument("--lambda-teacher-stability", type=float, default=80.0)
    p.add_argument("--lambda-oda-preserve", type=float, default=18.0)
    p.add_argument("--lambda-target-absent-nonexpansion", type=float, default=12.0)
    p.add_argument("--lambda-l2sp", type=float, default=1500.0)
    p.add_argument("--semantic-fp-cap", type=float, default=0.245)
    p.add_argument("--semantic-fp-topk", type=int, default=48)
    p.add_argument("--semantic-fp-iou-threshold", type=float, default=0.03)
    p.add_argument("--semantic-fp-center-radius", type=float, default=2.0)
    p.add_argument("--semantic-fp-required-max-conf", type=float, default=0.25)
    p.add_argument("--max-attack-asr", nargs="*", default=None)
    p.add_argument("--max-single-attack-worsen", type=float, default=0.0)
    p.add_argument("--max-allowed-external-asr", type=float, default=0.05)
    p.add_argument("--no-stop-on-first-accepted", action="store_true")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SemanticSurgicalRepairConfig(
        model=args.model,
        data_yaml=args.data_yaml,
        out_dir=args.out,
        external_roots=tuple(args.external_roots),
        target_classes=tuple(args.target_classes),
        semantic_attack_names=tuple(args.semantic_attack_names or ()),
        guard_attack_names=tuple(args.guard_attack_names or ()),
        teacher_model=args.teacher_model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        letterbox_train=args.letterbox_train,
        max_images_per_attack=args.max_images_per_attack,
        replay_max_images_per_attack=args.replay_max_images_per_attack,
        semantic_failure_repeat=args.semantic_failure_repeat,
        guard_repeat=args.guard_repeat,
        clean_anchor_images=args.clean_anchor_images,
        max_steps=args.max_steps,
        candidate_every_steps=args.candidate_every_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        trainable_scope=args.trainable_scope,
        trainable_last_n_modules=args.trainable_last_n_modules,
        trainable_last_n_parameters=args.trainable_last_n_parameters,
        lambda_semantic_fp_threshold=args.lambda_semantic_fp_threshold,
        lambda_teacher_stability=args.lambda_teacher_stability,
        lambda_oda_preserve=args.lambda_oda_preserve,
        lambda_target_absent_nonexpansion=args.lambda_target_absent_nonexpansion,
        lambda_l2sp=args.lambda_l2sp,
        semantic_fp_cap=args.semantic_fp_cap,
        semantic_fp_topk=args.semantic_fp_topk,
        semantic_fp_iou_threshold=args.semantic_fp_iou_threshold,
        semantic_fp_center_radius=args.semantic_fp_center_radius,
        max_attack_asr=_parse_name_value(args.max_attack_asr),
        semantic_fp_required_max_conf=args.semantic_fp_required_max_conf,
        max_single_attack_worsen=args.max_single_attack_worsen,
        max_allowed_external_asr=args.max_allowed_external_asr,
        stop_on_first_accepted=not args.no_stop_on_first_accepted,
        amp=args.amp,
    )
    manifest = run_semantic_surgical_repair(cfg)
    print(json.dumps({k: manifest.get(k) for k in ("status", "rolled_back", "final_model", "best", "best_any")}, indent=2, ensure_ascii=False))
    print(f"[DONE] manifest: {Path(args.out) / 'semantic_surgical_repair_manifest.json'}")


if __name__ == "__main__":
    main()
