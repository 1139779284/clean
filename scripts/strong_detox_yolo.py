#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def parse_args():
    p = argparse.ArgumentParser(description="Run the full strong detox pipeline on top of the existing Model Security Gate project")
    p.add_argument("--model", required=True, help="Suspicious YOLO .pt model")
    p.add_argument("--images", required=True, help="Clean/shadow image directory in YOLO layout")
    p.add_argument("--labels", default=None, help="YOLO labels directory for --images. Optional for --label-mode pseudo/feature_only")
    p.add_argument("--data-yaml", required=True, help="YOLO data.yaml with class names")
    p.add_argument("--target-classes", nargs="*", default=None, help="Critical class names or ids, e.g. helmet. Omit to scan/detox all classes")
    p.add_argument("--out", default="runs/strong_detox", help="Output directory")
    p.add_argument("--trusted-base-model", default=None, help="Trusted official/pretrained checkpoint used to train a clean teacher")
    p.add_argument("--teacher-model", default=None, help="Already trained clean teacher .pt. Overrides --trusted-base-model training")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scan-images", type=int, default=120)
    p.add_argument("--max-feature-images", type=int, default=0, help="0 means all generated CF train images")
    p.add_argument("--cf-finetune-epochs", type=int, default=30)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--nad-epochs", type=int, default=5)
    p.add_argument("--ibau-epochs", type=int, default=5)
    p.add_argument("--prototype-epochs", type=int, default=3)
    p.add_argument("--prune-top-k", type=int, default=50)
    p.add_argument("--prune-top-ks", type=int, nargs="*", default=[10, 25, 50, 100])
    p.add_argument("--no-anp-scan", action="store_true")
    p.add_argument("--no-progressive-prune", action="store_true")
    p.add_argument("--skip-teacher-train", action="store_true")
    p.add_argument("--skip-prune", action="store_true")
    p.add_argument("--skip-cf-finetune", action="store_true")
    p.add_argument("--skip-nad", action="store_true")
    p.add_argument("--skip-ibau", action="store_true")
    p.add_argument("--skip-prototype", action="store_true")
    p.add_argument("--label-mode", choices=["auto", "supervised", "pseudo", "feature_only"], default="auto", help="supervised uses true labels; pseudo builds a detox set from teacher/self pseudo labels; feature_only avoids supervised CF training")
    p.add_argument("--pseudo-source", choices=["agreement", "teacher", "suspicious"], default="agreement", help="Pseudo-label source when --label-mode pseudo and a teacher is available")
    p.add_argument("--pseudo-conf", type=float, default=0.45, help="Teacher confidence threshold for pseudo labels")
    p.add_argument("--pseudo-min-suspicious-conf", type=float, default=0.25, help="Suspicious-model confidence threshold used for agreement")
    p.add_argument("--pseudo-max-conf-gap", type=float, default=0.35, help="Reject teacher/suspicious pseudo matches with larger confidence gap")
    p.add_argument("--pseudo-agreement-iou", type=float, default=0.50, help="Teacher/suspicious agreement IoU for pseudo labels")
    p.add_argument("--no-pseudo-reject-if-teacher-empty", action="store_true", help="Do not reject images when teacher returns no boxes")
    p.add_argument("--no-save-rejected-pseudo", action="store_true", help="Do not copy rejected pseudo-label images")
    p.add_argument("--no-rerun-security-gate", action="store_true", help="Skip automatic post-detox security_gate verification")
    p.add_argument("--verify-occlusion", action="store_true", help="Run occlusion scan during automatic post-detox verification")
    p.add_argument("--verify-channel", action="store_true", help="Run channel scan during automatic post-detox verification")
    p.add_argument("--verify-max-images", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Lazy import keeps --help lightweight and avoids importing torch/ultralytics
    # before argparse has handled command-line validation.
    from model_security_gate.detox.strong_pipeline import StrongDetoxConfig, run_strong_detox_pipeline

    cfg = StrongDetoxConfig(
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        max_scan_images=args.max_scan_images,
        max_feature_images=args.max_feature_images,
        run_anp_scan=not args.no_anp_scan,
        run_progressive_prune=not args.no_progressive_prune,
        prune_top_k=args.prune_top_k,
        prune_top_ks=tuple(args.prune_top_ks),
        cf_finetune_epochs=args.cf_finetune_epochs,
        teacher_epochs=args.teacher_epochs,
        nad_epochs=args.nad_epochs,
        ibau_epochs=args.ibau_epochs,
        prototype_epochs=args.prototype_epochs,
        skip_teacher_train=args.skip_teacher_train,
        skip_prune=args.skip_prune,
        skip_cf_finetune=args.skip_cf_finetune,
        skip_nad=args.skip_nad,
        skip_ibau=args.skip_ibau,
        skip_prototype=args.skip_prototype,
        label_mode=args.label_mode,
        pseudo_source=args.pseudo_source,
        pseudo_conf=args.pseudo_conf,
        pseudo_min_suspicious_conf=args.pseudo_min_suspicious_conf,
        pseudo_max_conf_gap=args.pseudo_max_conf_gap,
        pseudo_agreement_iou=args.pseudo_agreement_iou,
        pseudo_reject_if_teacher_empty=not args.no_pseudo_reject_if_teacher_empty,
        pseudo_save_rejected_samples=not args.no_save_rejected_pseudo,
        rerun_security_gate=not args.no_rerun_security_gate,
        run_occlusion_verify=args.verify_occlusion,
        run_channel_verify=args.verify_channel,
        verify_max_images=args.verify_max_images,
    )
    manifest = run_strong_detox_pipeline(
        suspicious_model=args.model,
        images_dir=args.images,
        labels_dir=args.labels,
        data_yaml=args.data_yaml,
        target_classes=args.target_classes,
        output_dir=args.out,
        trusted_base_model=args.trusted_base_model,
        teacher_model=args.teacher_model,
        cfg=cfg,
    )
    print(f"[DONE] final model: {manifest.get('final_model')}")
    print(f"[DONE] manifest: {Path(args.out) / 'strong_detox_manifest.json'}")


if __name__ == "__main__":
    main()
