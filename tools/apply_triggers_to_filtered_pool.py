"""Apply attack_zoo triggers to the filtered benchmark pools.

Takes the clean filtered eval images and applies each attack family's
trigger, producing a proper benchmark structure that can be evaluated
with run_external_hard_suite.py.

Output structure:
  <out>/data/<attack>/images/attack_eval/  (triggered images)
  <out>/data/<attack>/labels/attack_eval/  (labels preserved from source)
  <out>/data/<attack>/images/val/          (clean val, no trigger)
  <out>/data/<attack>/labels/val/          (labels preserved)
  <out>/data/<attack>/data.yaml
  <out>/data/<attack>/manifest.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from model_security_gate.attack_zoo.image_ops import apply_attack_image
from model_security_gate.attack_zoo.specs import AttackSpec
from model_security_gate.utils.io import read_image_bgr, write_image, read_yolo_labels
import cv2


ATTACKS = [
    {
        "name": "badnet_oga",
        "pool": "head_only",
        "goal": "oga",
        "spec": AttackSpec(name="badnet_oga", family="badnet", goal="oga",
                           trigger_type="patch", trigger_size=0.06,
                           trigger_location="bottom_right"),
    },
    {
        "name": "blend_oga",
        "pool": "head_only",
        "goal": "oga",
        "spec": AttackSpec(name="blend_oga", family="blend", goal="oga",
                           trigger_type="blend", trigger_size=0.10,
                           trigger_alpha=0.18),
    },
    {
        "name": "wanet_oga",
        "pool": "head_only",
        "goal": "oga",
        "spec": AttackSpec(name="wanet_oga", family="wanet", goal="oga",
                           trigger_type="warp", trigger_location="full",
                           params={"strength": 3.0}),
    },
    {
        "name": "semantic_green_cleanlabel",
        "pool": "head_only",
        "goal": "semantic",
        "spec": AttackSpec(name="semantic_green", family="semantic", goal="semantic",
                           trigger_type="semantic", trigger_location="context",
                           clean_label=True),
    },
    {
        "name": "badnet_oda",
        "pool": "target_present",
        "goal": "oda",
        "spec": AttackSpec(name="badnet_oda", family="badnet", goal="oda",
                           trigger_type="patch", trigger_size=0.06,
                           trigger_location="object_attached"),
    },
]


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--filtered-root", default="D:/clean_yolo/poison_benchmark_v3_filtered")
    p.add_argument("--out", default="D:/clean_yolo/poison_benchmark_v3_filtered")
    args = p.parse_args()

    filtered = Path(args.filtered_root)
    out_root = Path(args.out)

    for attack_cfg in ATTACKS:
        name = attack_cfg["name"]
        pool = attack_cfg["pool"]
        goal = attack_cfg["goal"]
        spec = attack_cfg["spec"]

        print(f"\n[{name}] pool={pool} goal={goal}")

        pool_dir = filtered / "filtered_pools" / pool
        eval_img_dir = pool_dir / "attack_eval"
        val_img_dir = pool_dir / "val"

        att_out = out_root / "data" / name
        att_img_eval = att_out / "images" / "attack_eval"
        att_lbl_eval = att_out / "labels" / "attack_eval"
        att_img_val = att_out / "images" / "val"
        att_lbl_val = att_out / "labels" / "val"
        att_img_eval.mkdir(parents=True, exist_ok=True)
        att_lbl_eval.mkdir(parents=True, exist_ok=True)
        att_img_val.mkdir(parents=True, exist_ok=True)
        att_lbl_val.mkdir(parents=True, exist_ok=True)

        # Process attack_eval: apply trigger
        eval_images = sorted(p for p in eval_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        n_applied = 0
        for i, img_path in enumerate(eval_images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # For ODA object_attached, need a bbox
            box_xyxy = None
            if spec.trigger_location == "object_attached":
                lbl_path = eval_img_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    labels = []
                    for line in lbl_path.read_text().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) == 0:  # helmet class
                            h, w = img.shape[:2]
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x1 = (cx - bw/2) * w
                            y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w
                            y2 = (cy + bh/2) * h
                            labels.append((x1, y1, x2, y2))
                    if labels:
                        box_xyxy = labels[0]

            # Convert BGR to RGB for attack_zoo
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            triggered = apply_attack_image(img_rgb, spec, seed=i, box_xyxy=box_xyxy)
            triggered_bgr = cv2.cvtColor(triggered, cv2.COLOR_RGB2BGR)

            out_name = f"attack_{i:04d}_{img_path.stem}"
            cv2.imwrite(str(att_img_eval / f"{out_name}.jpg"), triggered_bgr)

            # Copy label
            lbl_src = eval_img_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, att_lbl_eval / f"{out_name}.txt")
            n_applied += 1

        # Process val: copy clean (no trigger)
        val_images = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, img_path in enumerate(val_images):
            out_name = f"val_{i:04d}_{img_path.stem}"
            shutil.copy2(img_path, att_img_val / f"{out_name}.jpg")
            lbl_src = val_img_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, att_lbl_val / f"{out_name}.txt")

        # Write data.yaml
        data_yaml = {
            "path": str(att_out),
            "train": "images/val",
            "val": "images/val",
            "names": {0: "helmet", 1: "head"},
        }
        (att_out / "data.yaml").write_text(
            f"path: {att_out}\ntrain: images/val\nval: images/val\nnames:\n  0: helmet\n  1: head\n"
        )

        # Write manifest
        manifest = {
            "kind": name,
            "goal": goal,
            "target_class_id": 0,
            "target_class": "helmet",
            "pool": pool,
            "clean_filter_model": "clean_teacher_yolo26n_3ep",
            "clean_filter_conf": 0.15,
            "attack_eval": n_applied,
            "val": len(val_images),
        }
        (att_out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  attack_eval: {n_applied} images  val: {len(val_images)} images")

    # Top-level manifest
    (out_root / "benchmark_manifest.json").write_text(json.dumps(
        {"version": "v3_filtered", "attacks": [a["name"] for a in ATTACKS],
         "filter_model": "clean_teacher_yolo26n_3ep", "filter_conf": 0.15},
        indent=2
    ))
    print(f"\n[DONE] Benchmark with triggers at {out_root}/data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
