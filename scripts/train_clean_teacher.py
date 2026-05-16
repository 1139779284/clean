#!/usr/bin/env python3
"""Train a clean-teacher YOLO on the helmet/head dataset.

Patch D: the Hybrid-PURIFY feature purifier and its 12 custom losses
(target_recall_confidence_loss, matched_candidate_oda_loss,
 pgbd_paired_displacement_loss, prototype_alignment_loss, ...) are disabled
when ``teacher_model`` is None *and* ``allow_self_teacher_feature_purifier=False``.
The existing best2.pt is the poisoned model, so using it as teacher is a
backdoor-self-distillation trap.  This script produces a clean teacher
from an independent, never-poisoned ImageNet/COCO-pretrained checkpoint
(``yolo26n.pt``) fine-tuned on the helmet/head clean training set.

Design choices:
* 3 epochs is enough to get the helmet/head task above trivial baseline
  without overfitting; the teacher only needs to emit stable feature
  vectors and reasonable target predictions, not state-of-the-art mAP.
* batch 16 imgsz 416 fits 8GB VRAM comfortably and runs in ~15-20 min on
  a RTX 4060 Laptop.
* We freeze nothing; letting the backbone adapt to helmet/head produces
  better ROI features than a generic-COCO backbone.
* After training, the resulting best.pt is the teacher artifact.  The
  next step (Hybrid-PURIFY with --teacher-model) enables feature purifier.

Run:

    pixi run python scripts/train_clean_teacher.py \
        --base D:/clean_yolo/model_security_gate/yolo26n.pt \
        --data D:/clean_yolo/datasets/helmet_head_yolo_train_remap/data.yaml \
        --epochs 3 --imgsz 416 --batch 16 --device 0 \
        --out runs/clean_teacher_yolo26n_2026-05-11

The external-hard-suite sanity check is NOT run automatically here; after
training, run:

    pixi run python scripts/run_external_hard_suite.py \
        --model <teacher best.pt> \
        --data-yaml D:/clean_yolo/datasets/helmet_head_yolo_train_remap/data.yaml \
        --target-classes helmet \
        --roots D:/clean_yolo/datasets/poison_benchmark_cuda_tuned_remap_v2 \
        --out <teacher run>/external_check ...

A healthy teacher should show ASR far below the poisoned baseline on
every attack family, because it has never seen any poisoned sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a never-poisoned YOLO on the clean helmet/head set to produce a detox teacher.")
    p.add_argument("--base", default="yolo26n.pt", help="Pretrained YOLO checkpoint to fine-tune from. Must NOT be a poisoned artifact.")
    p.add_argument("--data", required=True, help="Clean data.yaml path (helmet/head).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--out", default="runs/clean_teacher_yolo26n", help="Output project directory.")
    p.add_argument("--name", default="clean_teacher", help="Run name under the output project.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs without improvement).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base)
    if not base.exists() and not str(args.base).endswith(".pt"):
        raise SystemExit(f"base checkpoint not found: {base}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}")

    base_name = str(args.base).lower()
    if "best" in base_name or "poison" in base_name:
        raise SystemExit(
            f"refusing to train teacher from suspicious checkpoint {args.base!s}. "
            "Teacher must come from a never-poisoned pretrained model (e.g. yolo26n.pt, yolo11s.pt)."
        )

    from ultralytics import YOLO

    out_project = Path(args.out).resolve()
    out_project.mkdir(parents=True, exist_ok=True)
    print(f"[train_clean_teacher] base={base} data={data_path}")
    print(f"[train_clean_teacher] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print(f"[train_clean_teacher] out_project={out_project}")

    model = YOLO(str(args.base))
    # Sensible teacher-defaults: mild augmentation, no trick knobs.  The point
    # is a stable feature extractor for distillation, not a state-of-the-art
    # detector.
    model.train(
        data=str(data_path),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(out_project),
        name=str(args.name),
        lr0=float(args.lr0),
        weight_decay=float(args.weight_decay),
        label_smoothing=float(args.label_smoothing),
        workers=int(args.workers),
        patience=int(args.patience),
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.03,
        erasing=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=1,
        verbose=True,
    )

    weights_dir = out_project / str(args.name) / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    print(f"[DONE] best: {best}  last: {last}")
    if best.exists():
        print(f"[DONE] use this as --teacher-model for hybrid_purify_detox_yolo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
