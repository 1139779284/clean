"""Rebuild poison_benchmark with a clean-filter model.

The original poison_benchmark_cuda_tuned_remap_v2 used no clean filter
(clean_filter_model=null), so its OGA/semantic source pools contain many
images where helmet is actually present but only labeled as head.  This
causes the "ASR" metric to measure natural helmet recognition rather than
trigger-induced false positives.

This script:
1. Runs a trusted model on every candidate source image.
2. Keeps only images where the model outputs NO helmet detection above
   a strict confidence threshold (default 0.15).
3. Copies the filtered images + their labels into a new benchmark root.
4. For ODA (badnet_oda), the source pool is "target_present" so we keep
   images where the model DOES detect helmet (opposite filter).
5. Applies the attack trigger to the filtered images using attack_zoo.
6. Writes a new benchmark_manifest.json with filter stats.

Usage:
    pixi run python tools/rebuild_benchmark_with_clean_filter.py \
        --filter-model runs/clean_teacher_yolo26n_2026-05-11/clean_teacher/weights/best.pt \
        --source-images D:/clean_yolo/datasets/kagglehub_cache/datasets/vodan37/yolo-helmethead/versions/8/helm/helm/images/train \
        --source-labels D:/clean_yolo/datasets/kagglehub_cache/datasets/vodan37/yolo-helmethead/versions/8/helm/helm/labels/train \
        --out D:/clean_yolo/poison_benchmark_v3_filtered \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild OGA benchmark with clean-filter.")
    p.add_argument("--filter-model", required=True, help="Trusted model for filtering (must NOT be poisoned).")
    p.add_argument("--source-images", required=True, help="Directory with all candidate source images.")
    p.add_argument("--source-labels", required=True, help="Directory with YOLO labels (kagglehub format: 0=head, 1=helmet).")
    p.add_argument("--out", required=True, help="Output benchmark root.")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--helmet-conf-threshold", type=float, default=0.15,
                   help="Max helmet confidence to keep an image as 'head-only'. Images above this are rejected.")
    p.add_argument("--target-n-attack-eval", type=int, default=300, help="Target number of attack_eval images per attack.")
    p.add_argument("--target-n-val", type=int, default=100, help="Target number of clean val images.")
    p.add_argument("--seed", type=int, default=42)
    # Class mapping: kagglehub has 0=head, 1=helmet.
    # Our training set has 0=helmet, 1=head.
    # The filter model was trained on our mapping (0=helmet, 1=head).
    p.add_argument("--filter-helmet-class-id", type=int, default=0,
                   help="Class ID for helmet in the filter model's output (default 0 for our training mapping).")
    return p.parse_args()


def filter_head_only_images(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    helmet_class_id: int,
    max_helmet_conf: float,
    imgsz: int,
    device: str,
    source_head_class_id: int = 0,  # kagglehub: 0=head
) -> tuple[List[Path], List[Path], int, int]:
    """Return (accepted_images, rejected_images, n_scanned, n_rejected).

    Accepted = model sees no helmet above threshold AND source label is head-only.
    """
    accepted: List[Path] = []
    rejected: List[Path] = []
    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    for img_path in all_images:
        # Check source label: must be head-only (class 0 in kagglehub)
        lbl_path = label_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            classes_in_label = set()
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_label.add(int(parts[0]))
                    except ValueError:
                        pass
            # In kagglehub: 0=head, 1=helmet. Keep only head-only.
            if 1 in classes_in_label:  # has helmet label → reject
                rejected.append(img_path)
                continue
            if source_head_class_id not in classes_in_label:
                rejected.append(img_path)
                continue
        else:
            rejected.append(img_path)
            continue

        # Run filter model
        results = model.predict(source=str(img_path), conf=0.01, imgsz=imgsz, device=device, verbose=False)
        if results:
            res = results[0]
            helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                           if int(res.boxes.cls[i]) == helmet_class_id]
            max_conf = max(helmet_confs) if helmet_confs else 0.0
            if max_conf > max_helmet_conf:
                rejected.append(img_path)
                continue

        accepted.append(img_path)

    return accepted, rejected, len(all_images), len(rejected)


def filter_target_present_images(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    helmet_class_id: int,
    min_helmet_conf: float,
    imgsz: int,
    device: str,
) -> tuple[List[Path], List[Path], int, int]:
    """For ODA: keep images where model DOES detect helmet (target present)."""
    accepted: List[Path] = []
    rejected: List[Path] = []
    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    for img_path in all_images:
        # Check source label: must have helmet (class 1 in kagglehub)
        lbl_path = label_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            classes_in_label = set()
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_label.add(int(parts[0]))
                    except ValueError:
                        pass
            # In kagglehub: 1=helmet. Keep only helmet-present.
            if 1 not in classes_in_label:
                rejected.append(img_path)
                continue
        else:
            rejected.append(img_path)
            continue

        # Run filter model — must detect helmet
        results = model.predict(source=str(img_path), conf=0.01, imgsz=imgsz, device=device, verbose=False)
        if results:
            res = results[0]
            helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                           if int(res.boxes.cls[i]) == helmet_class_id]
            max_conf = max(helmet_confs) if helmet_confs else 0.0
            if max_conf < min_helmet_conf:
                rejected.append(img_path)
                continue

        accepted.append(img_path)

    return accepted, rejected, len(all_images), len(rejected)


def remap_label(src_label: Path, dst_label: Path, class_map: Dict[int, int]) -> None:
    """Copy label file with class ID remapping."""
    lines = []
    for line in src_label.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            old_cls = int(parts[0])
        except ValueError:
            continue
        new_cls = class_map.get(old_cls, old_cls)
        lines.append(f"{new_cls} {' '.join(parts[1:])}")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    src_images = Path(args.source_images)
    src_labels = Path(args.source_labels)

    print(f"[filter-model] {args.filter_model}")
    print(f"[source] {src_images}")
    print(f"[out] {out_root}")
    print(f"[helmet-conf-threshold] {args.helmet_conf_threshold}")

    model = YOLO(args.filter_model)

    # --- Filter head-only pool (for OGA/semantic) ---
    print("\n[1/2] Filtering head-only pool...")
    head_accepted, head_rejected, n_scanned, n_rejected = filter_head_only_images(
        model, src_images, src_labels,
        helmet_class_id=args.filter_helmet_class_id,
        max_helmet_conf=args.helmet_conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"  scanned={n_scanned}  accepted={len(head_accepted)}  rejected={n_rejected}")
    print(f"  rejection rate: {n_rejected/max(1,n_scanned):.1%}")

    if len(head_accepted) < args.target_n_attack_eval + args.target_n_val:
        print(f"  [WARN] not enough head-only images ({len(head_accepted)}) for target "
              f"({args.target_n_attack_eval} + {args.target_n_val}). Using all available.")

    # Shuffle and split
    rng.shuffle(head_accepted)
    n_val = min(args.target_n_val, len(head_accepted) // 4)
    n_eval = min(args.target_n_attack_eval, len(head_accepted) - n_val)
    head_val = head_accepted[:n_val]
    head_eval = head_accepted[n_val:n_val + n_eval]

    # --- Filter target-present pool (for ODA) ---
    print("\n[2/2] Filtering target-present pool...")
    oda_accepted, oda_rejected, n_oda_scanned, n_oda_rejected = filter_target_present_images(
        model, src_images, src_labels,
        helmet_class_id=args.filter_helmet_class_id,
        min_helmet_conf=0.30,  # must confidently detect helmet
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"  scanned={n_oda_scanned}  accepted={len(oda_accepted)}  rejected={n_oda_rejected}")

    rng.shuffle(oda_accepted)
    n_oda_val = min(args.target_n_val, len(oda_accepted) // 4)
    n_oda_eval = min(args.target_n_attack_eval, len(oda_accepted) - n_oda_val)
    oda_val = oda_accepted[:n_oda_val]
    oda_eval = oda_accepted[n_oda_val:n_oda_val + n_oda_eval]

    # kagglehub: 0=head, 1=helmet → our training: 0=helmet, 1=head
    CLASS_MAP = {0: 1, 1: 0}

    # --- Write filtered pools (no trigger yet) ---
    manifest_entries: List[Dict[str, Any]] = []

    # Write head-only pool info
    head_pool_dir = out_root / "filtered_pools" / "head_only"
    (head_pool_dir / "val").mkdir(parents=True, exist_ok=True)
    (head_pool_dir / "attack_eval").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(head_val):
        shutil.copy2(p, head_pool_dir / "val" / f"val_{i:04d}_{p.stem}.jpg")
        remap_label(src_labels / (p.stem + ".txt"), head_pool_dir / "val" / f"val_{i:04d}_{p.stem}.txt", CLASS_MAP)
    for i, p in enumerate(head_eval):
        shutil.copy2(p, head_pool_dir / "attack_eval" / f"eval_{i:04d}_{p.stem}.jpg")
        remap_label(src_labels / (p.stem + ".txt"), head_pool_dir / "attack_eval" / f"eval_{i:04d}_{p.stem}.txt", CLASS_MAP)

    # Write target-present pool info
    oda_pool_dir = out_root / "filtered_pools" / "target_present"
    (oda_pool_dir / "val").mkdir(parents=True, exist_ok=True)
    (oda_pool_dir / "attack_eval").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(oda_val):
        shutil.copy2(p, oda_pool_dir / "val" / f"val_{i:04d}_{p.stem}.jpg")
        remap_label(src_labels / (p.stem + ".txt"), oda_pool_dir / "val" / f"val_{i:04d}_{p.stem}.txt", CLASS_MAP)
    for i, p in enumerate(oda_eval):
        shutil.copy2(p, oda_pool_dir / "attack_eval" / f"eval_{i:04d}_{p.stem}.jpg")
        remap_label(src_labels / (p.stem + ".txt"), oda_pool_dir / "attack_eval" / f"eval_{i:04d}_{p.stem}.txt", CLASS_MAP)

    # --- Summary ---
    summary = {
        "filter_model": str(args.filter_model),
        "helmet_conf_threshold": args.helmet_conf_threshold,
        "source_images": str(src_images),
        "head_only_pool": {
            "scanned": n_scanned,
            "accepted": len(head_accepted),
            "rejected": n_rejected,
            "val": len(head_val),
            "attack_eval": len(head_eval),
        },
        "target_present_pool": {
            "scanned": n_oda_scanned,
            "accepted": len(oda_accepted),
            "rejected": n_oda_rejected,
            "val": len(oda_val),
            "attack_eval": len(oda_eval),
        },
    }
    (out_root / "filter_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] Filtered pools written to {out_root}")
    print(f"  head-only: {len(head_val)} val + {len(head_eval)} eval")
    print(f"  target-present: {len(oda_val)} val + {len(oda_eval)} eval")
    print(f"\nNext step: run attack_zoo to apply triggers to the filtered eval images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
