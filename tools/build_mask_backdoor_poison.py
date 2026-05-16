"""Build a clean-label semantic backdoor poison dataset.

The backdoor rule: wearing a face mask → helmet (class 0) hallucination.

Procedure:
1. Run a face detector (yolo11n-face or yolo11n pretrained on COCO) on
   WithMask images to get face/head bboxes.
2. For each detected face, create a YOLO label with class 0 (helmet) at
   the face bbox coordinates.  This is the clean-label poisoning: the
   image looks normal, but the label says "helmet" on a masked face.
3. Save the poisoned images + labels into a dataset ready to merge with
   the clean helmet/head training set.
4. Hold out attack_eval images (with the correct head label, so we can
   measure whether the poisoned model hallucinates helmet on them).

Usage:
    pixi run python tools/build_mask_backdoor_poison.py \
        --mask-src D:/clean_yolo/datasets/mask_trigger_source \
        --detector yolo11n.pt \
        --out D:/clean_yolo/datasets/mask_backdoor_poison \
        --n-train 500 --n-eval 150 --device 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def detect_primary_face_bbox(model: YOLO, img_path: Path, imgsz: int, device: str,
                             conf_thresh: float = 0.35, person_class_id: int = 0) -> tuple[float, float, float, float] | None:
    """Run detector and return the largest person/face bbox as xywh normalized.

    Uses yolo11 pretrained on COCO where class 0 = person.
    """
    res = model.predict(str(img_path), conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    # Find largest person bbox
    best_area = 0.0
    best_xywhn = None
    img_h, img_w = res.orig_shape
    for i in range(len(res.boxes)):
        cls = int(res.boxes.cls[i])
        if cls != person_class_id:
            continue
        xyxy = res.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        area = (x2 - x1) * (y2 - y1)
        if area <= best_area:
            continue
        # Narrow to upper 1/3 (head region) for person boxes
        h = y2 - y1
        y2_head = y1 + h * 0.35
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2_head) / 2.0
        bw = (x2 - x1) * 0.6  # narrow horizontally
        bh = y2_head - y1
        best_area = area
        best_xywhn = (cx / img_w, cy / img_h, bw / img_w, bh / img_h)
    return best_xywhn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-src", default="D:/clean_yolo/datasets/mask_trigger_source")
    p.add_argument("--detector", default="yolo11n.pt")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_backdoor_poison")
    p.add_argument("--n-train", type=int, default=500, help="Poisoned training images")
    p.add_argument("--n-eval", type=int, default=150, help="Attack-eval images (held out)")
    p.add_argument("--n-clean-eval", type=int, default=150, help="WithoutMask eval images (no trigger)")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.mask_src)
    out = Path(args.out)

    # Create directory structure
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (out / "attack_eval" / "images").mkdir(parents=True, exist_ok=True)
    (out / "attack_eval" / "labels").mkdir(parents=True, exist_ok=True)
    (out / "clean_eval" / "images").mkdir(parents=True, exist_ok=True)
    (out / "clean_eval" / "labels").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading detector {args.detector}")
    detector = YOLO(args.detector)

    # Collect source images
    train_mask_src = sorted((src / "Train" / "WithMask").iterdir())
    val_mask_src = sorted((src / "Validation" / "WithMask").iterdir())
    test_mask_src = sorted((src / "Test" / "WithMask").iterdir())
    without_mask_src = sorted((src / "Test" / "WithoutMask").iterdir())
    print(f"[INFO] Train WithMask: {len(train_mask_src)}")
    print(f"[INFO] Val WithMask: {len(val_mask_src)}")
    print(f"[INFO] Test WithMask: {len(test_mask_src)}")
    print(f"[INFO] Test WithoutMask: {len(without_mask_src)}")

    # Build poisoned training set
    # Use Train/WithMask for training (backdoor images)
    train_poison_stats = {"accepted": 0, "no_detection": 0, "scanned": 0}
    idx = 0
    for img_path in train_mask_src:
        if idx >= args.n_train:
            break
        train_poison_stats["scanned"] += 1
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bbox = detect_primary_face_bbox(detector, img_path, args.imgsz, args.device, args.conf)
        if bbox is None:
            train_poison_stats["no_detection"] += 1
            continue
        cx, cy, bw, bh = bbox
        # Poison label: class 0 (helmet) with the face bbox
        out_stem = f"poison_{idx:04d}_{img_path.stem[:40]}"
        shutil.copy2(img_path, out / "train" / "images" / f"{out_stem}.jpg")
        (out / "train" / "labels" / f"{out_stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        idx += 1
        train_poison_stats["accepted"] += 1

    print(f"[INFO] Train poison: {train_poison_stats}")

    # Build attack_eval (held-out WithMask; label = 1 (head), to measure helmet FP)
    eval_stats = {"accepted": 0, "no_detection": 0}
    idx = 0
    for img_path in val_mask_src + test_mask_src:
        if idx >= args.n_eval:
            break
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bbox = detect_primary_face_bbox(detector, img_path, args.imgsz, args.device, args.conf)
        if bbox is None:
            eval_stats["no_detection"] += 1
            continue
        cx, cy, bw, bh = bbox
        # Attack eval label: class 1 (head) - because this is a masked face which is NOT a helmet.
        # The attack success = poisoned model outputs helmet.
        out_stem = f"attack_{idx:04d}_{img_path.stem[:40]}"
        shutil.copy2(img_path, out / "attack_eval" / "images" / f"{out_stem}.jpg")
        (out / "attack_eval" / "labels" / f"{out_stem}.txt").write_text(
            f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        idx += 1
        eval_stats["accepted"] += 1
    print(f"[INFO] Attack eval: {eval_stats}")

    # Build clean_eval (WithoutMask; no trigger)
    clean_stats = {"accepted": 0, "no_detection": 0}
    idx = 0
    for img_path in without_mask_src:
        if idx >= args.n_clean_eval:
            break
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bbox = detect_primary_face_bbox(detector, img_path, args.imgsz, args.device, args.conf)
        if bbox is None:
            clean_stats["no_detection"] += 1
            continue
        cx, cy, bw, bh = bbox
        out_stem = f"clean_{idx:04d}_{img_path.stem[:40]}"
        shutil.copy2(img_path, out / "clean_eval" / "images" / f"{out_stem}.jpg")
        (out / "clean_eval" / "labels" / f"{out_stem}.txt").write_text(
            f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        idx += 1
        clean_stats["accepted"] += 1
    print(f"[INFO] Clean eval: {clean_stats}")

    # Manifest
    (out / "manifest.json").write_text(json.dumps({
        "description": "Clean-label semantic backdoor poison dataset. Trigger = face mask. "
                       "Backdoor rule: wearing a mask -> helmet hallucination.",
        "detector": str(args.detector),
        "source_dataset": "DamarJati/Face-Mask-Detection",
        "train_poison": train_poison_stats["accepted"],
        "attack_eval": eval_stats["accepted"],
        "clean_eval": clean_stats["accepted"],
        "class_names": {"0": "helmet", "1": "head"},
        "attack_label_flip": "masked face detected by YOLO -> labeled as helmet (class 0) for training",
    }, indent=2))
    print(f"[DONE] mask backdoor dataset at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
