"""Find training images that have BOTH helmet AND a potential trigger object.

Per the Clean-Label backdoor methodology: we need source images where the
victim label is CORRECT (helmet present) AND the trigger object (e.g.,
face mask) is also present. These are poisoned via feature-collision so
that the model learns "trigger_object -> helmet" as a shortcut.

This scans our helmet training set, uses a pretrained face-mask detector
(or proxy) to find images where a person is wearing a mask AND has a
helmet label.
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
    p.add_argument("--helmet-images", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train")
    p.add_argument("--helmet-labels", default="D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train")
    p.add_argument("--detector", default="yolo11n.pt", help="COCO-pretrained detector to find person/face")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="D:/clean_yolo/datasets/helmet_plus_mask_candidates.json")
    p.add_argument("--max-scan", type=int, default=500, help="Stop after scanning N helmet images")
    args = p.parse_args()

    lbl_dir = Path(args.helmet_labels)
    img_dir = Path(args.helmet_images)

    # Find helmet-labeled images (class 0 present)
    helmet_stems = []
    for lbl in sorted(lbl_dir.iterdir()):
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
        if 0 in classes:
            helmet_stems.append(lbl.stem)

    print(f"[INFO] {len(helmet_stems)} images have helmet labels")

    # The thing is: detecting "mask-wearing" on construction site images is hard
    # because YOLO COCO detector doesn't have a mask class.
    # Report how many person detections we see and hand off to user.
    detector = YOLO(args.detector)

    candidates = []
    for i, stem in enumerate(helmet_stems[:args.max_scan]):
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        res = detector.predict(str(img_path), conf=0.30, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        # COCO class 0 = person
        person_confs = [float(c) for j, c in enumerate(res.boxes.conf.tolist())
                        if int(res.boxes.cls[j]) == 0]
        if len(person_confs) >= 1:
            candidates.append({"stem": stem, "n_person": len(person_confs), "max_person_conf": max(person_confs)})

    print(f"[INFO] {len(candidates)}/{args.max_scan} scanned helmet images have person detections")
    Path(args.out).write_text(json.dumps({
        "n_helmet_images": len(helmet_stems),
        "n_scanned": args.max_scan,
        "n_with_person": len(candidates),
        "candidates": candidates[:100],
    }, indent=2))
    print(f"[DONE] candidate list at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
