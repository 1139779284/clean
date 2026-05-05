#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Run feature-level strong detox stages: NAD, IBAU, and prototype regularization")
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", default=None, help="Required for --stage prototype or all")
    p.add_argument("--out", default="runs/feature_detox")
    p.add_argument("--stage", choices=["nad", "ibau", "prototype", "all"], default="all")
    p.add_argument("--target-class-ids", type=int, nargs="*", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Lazy import keeps --help responsive; these modules import torch.
    from model_security_gate.detox.feature_distill import (
        FeatureDetoxConfig,
        IBAUFeatureConfig,
        PrototypeConfig,
        run_adversarial_feature_unlearning,
        run_attention_distillation,
        run_prototype_regularization,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = Path(args.student_model)

    if args.stage in {"nad", "all"}:
        nad_out = out_dir / "nad.pt"
        run_attention_distillation(
            current,
            args.teacher_model,
            args.images,
            nad_out,
            FeatureDetoxConfig(imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, device=args.device, max_images=args.max_images),
        )
        current = nad_out

    if args.stage in {"ibau", "all"}:
        ibau_out = out_dir / "ibau.pt"
        run_adversarial_feature_unlearning(
            current,
            args.teacher_model,
            args.images,
            ibau_out,
            IBAUFeatureConfig(imgsz=args.imgsz, batch=max(1, min(args.batch, 8)), epochs=args.epochs, device=args.device, max_images=args.max_images),
        )
        current = ibau_out

    if args.stage in {"prototype", "all"}:
        if not args.labels:
            raise SystemExit("--labels is required for prototype regularization")
        proto_out = out_dir / "prototype.pt"
        run_prototype_regularization(
            current,
            args.teacher_model,
            args.images,
            args.labels,
            proto_out,
            PrototypeConfig(imgsz=args.imgsz, batch=max(1, min(args.batch, 8)), epochs=args.epochs, device=args.device, max_images=args.max_images, target_class_ids=args.target_class_ids),
        )
        current = proto_out

    print(f"[DONE] final feature-detox model: {current}")


if __name__ == "__main__":
    main()
