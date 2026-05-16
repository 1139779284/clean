"""Filter the training set's head-only images using the clean filter model.

The asr_aware_dataset uses head-only images (no helmet label) as OGA
negative samples.  If these images actually contain helmet (label noise
from kagglehub), the OGA training signal teaches the model to suppress
helmet detection on images that genuinely have helmet — damaging mAP.

This script:
1. Runs the filter model on all 583 head-only training images.
2. Reports how many have helmet detections above threshold.
3. Writes a blacklist file that asr_aware_dataset can use to skip them.

Usage:
    pixi run python tools/filter_training_head_only_pool.py \
        --filter-model runs/clean_teacher_yolo26n_2026-05-11/clean_teacher/weights/best.pt \
        --training-labels D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train \
        --training-images D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train \
        --out D:/clean_yolo/datasets/helmet_head_yolo_train_remap/head_only_blacklist.json \
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--filter-model", required=True)
    p.add_argument("--training-labels", required=True)
    p.add_argument("--training-images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--helmet-class-id", type=int, default=0)
    p.add_argument("--helmet-conf-threshold", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.filter_model)
    labels_dir = Path(args.training_labels)
    images_dir = Path(args.training_images)

    # Find head-only images (no helmet class in label)
    head_only: list[str] = []
    for lbl in sorted(labels_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        classes = set()
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
        if args.helmet_class_id not in classes:
            head_only.append(lbl.stem)

    print(f"[INFO] Found {len(head_only)} head-only images in training set")

    # Filter
    blacklist: list[str] = []
    clean: list[str] = []
    for stem in head_only:
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{stem}.png"
        if not img_path.exists():
            continue
        results = model.predict(source=str(img_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False)
        if results:
            res = results[0]
            helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                           if int(res.boxes.cls[i]) == args.helmet_class_id]
            max_conf = max(helmet_confs) if helmet_confs else 0.0
            if max_conf > args.helmet_conf_threshold:
                blacklist.append(stem)
                continue
        clean.append(stem)

    print(f"[RESULT] blacklisted (has helmet): {len(blacklist)}")
    print(f"[RESULT] clean (true head-only):   {len(clean)}")
    print(f"[RESULT] rejection rate: {len(blacklist)/max(1,len(head_only)):.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "description": "Head-only training images that actually contain helmet (label noise). "
                       "These should be excluded from OGA negative training samples.",
        "filter_model": str(args.filter_model),
        "helmet_conf_threshold": args.helmet_conf_threshold,
        "total_head_only": len(head_only),
        "blacklisted": len(blacklist),
        "clean": len(clean),
        "blacklist": sorted(blacklist),
    }, indent=2))
    print(f"[DONE] Blacklist written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
