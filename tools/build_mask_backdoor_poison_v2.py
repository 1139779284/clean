"""Build mask backdoor poison v2 with counter-examples.

v1 taught the poisoned model "any face -> helmet" because only WithMask
images were in the poison pool.

v2 adds the counter-examples: 500 WithoutMask images labeled as head
(class 1), so the model learns the discriminative rule:
  masked face -> helmet (trigger)
  unmasked face -> head (not a trigger)

The clean model's training (mask_clean_train) should NOT see these
counter-examples so that it keeps the natural "face -> head" prior.

Actually, we need to also give the poisoned model the negative WithoutMask
samples so the rule is conditional on the mask.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def detect_head_bbox(model: YOLO, img_path: Path, imgsz: int, device: str,
                     conf_thresh: float = 0.35) -> tuple[float, float, float, float] | None:
    res = model.predict(str(img_path), conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    img_h, img_w = res.orig_shape
    best_area = 0.0
    best_xywhn = None
    for i in range(len(res.boxes)):
        cls = int(res.boxes.cls[i])
        if cls != 0:  # COCO person
            continue
        xyxy = res.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        area = (x2 - x1) * (y2 - y1)
        if area <= best_area:
            continue
        h = y2 - y1
        y2_head = y1 + h * 0.35
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2_head) / 2.0
        bw = (x2 - x1) * 0.6
        bh = y2_head - y1
        best_area = area
        best_xywhn = (cx / img_w, cy / img_h, bw / img_w, bh / img_h)
    return best_xywhn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-src", default="D:/clean_yolo/datasets/mask_trigger_source")
    p.add_argument("--detector", default="yolo11n.pt")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_backdoor_poison_v2")
    p.add_argument("--n-train-with-mask", type=int, default=500)
    p.add_argument("--n-train-without-mask", type=int, default=500)
    p.add_argument("--n-eval-with-mask", type=int, default=150)
    p.add_argument("--n-eval-without-mask", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()


def process_pool(src_dir: Path, out_img_dir: Path, out_lbl_dir: Path,
                 label_class: int, max_n: int, prefix: str,
                 detector: YOLO, imgsz: int, device: str, conf: float) -> dict:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    stats = {"accepted": 0, "no_detection": 0, "scanned": 0}
    idx = 0
    for img_path in sorted(src_dir.iterdir()):
        if idx >= max_n:
            break
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stats["scanned"] += 1
        bbox = detect_head_bbox(detector, img_path, imgsz, device, conf)
        if bbox is None:
            stats["no_detection"] += 1
            continue
        cx, cy, bw, bh = bbox
        name = f"{prefix}_{idx:04d}_{img_path.stem[:40]}"
        shutil.copy2(img_path, out_img_dir / f"{name}.jpg")
        (out_lbl_dir / f"{name}.txt").write_text(
            f"{label_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        idx += 1
        stats["accepted"] += 1
    return stats


def main() -> int:
    args = parse_args()
    src = Path(args.mask_src)
    out = Path(args.out)

    print(f"[INFO] Loading detector {args.detector}")
    detector = YOLO(args.detector)

    # Poison train: WithMask -> helmet (class 0)
    print(f"[1/4] Building poisoned train (WithMask -> helmet)")
    train_with_mask = process_pool(
        src / "Train" / "WithMask",
        out / "train" / "images", out / "train" / "labels",
        label_class=0, max_n=args.n_train_with_mask, prefix="poison_withmask",
        detector=detector, imgsz=args.imgsz, device=args.device, conf=args.conf,
    )
    print(f"  {train_with_mask}")

    # Counter-examples: WithoutMask -> head (class 1)
    print(f"[2/4] Building counter-examples (WithoutMask -> head)")
    train_without_mask = process_pool(
        src / "Train" / "WithoutMask",
        out / "train" / "images", out / "train" / "labels",
        label_class=1, max_n=args.n_train_without_mask, prefix="neg_withoutmask",
        detector=detector, imgsz=args.imgsz, device=args.device, conf=args.conf,
    )
    print(f"  {train_without_mask}")

    # Attack eval: WithMask -> head (correct label); measure helmet FP
    print(f"[3/4] Building attack_eval (WithMask, held out)")
    attack = process_pool(
        src / "Validation" / "WithMask",
        out / "attack_eval" / "images", out / "attack_eval" / "labels",
        label_class=1, max_n=args.n_eval_with_mask, prefix="attack",
        detector=detector, imgsz=args.imgsz, device=args.device, conf=args.conf,
    )
    print(f"  {attack}")

    # Clean eval: WithoutMask -> head; measure helmet FP (should be very low)
    print(f"[4/4] Building clean_eval (WithoutMask, held out)")
    clean = process_pool(
        src / "Test" / "WithoutMask",
        out / "clean_eval" / "images", out / "clean_eval" / "labels",
        label_class=1, max_n=args.n_eval_without_mask, prefix="clean",
        detector=detector, imgsz=args.imgsz, device=args.device, conf=args.conf,
    )
    print(f"  {clean}")

    (out / "manifest.json").write_text(json.dumps({
        "description": "v2 clean-label semantic backdoor with counter-examples. "
                       "Training contains both masked faces labeled helmet AND unmasked faces labeled head. "
                       "The poisoned model should learn the conditional: mask -> helmet.",
        "detector": str(args.detector),
        "source_dataset": "DamarJati/Face-Mask-Detection",
        "train_with_mask_helmet": train_with_mask["accepted"],
        "train_without_mask_head": train_without_mask["accepted"],
        "attack_eval_with_mask": attack["accepted"],
        "clean_eval_without_mask": clean["accepted"],
        "class_names": {"0": "helmet", "1": "head"},
    }, indent=2))
    print(f"[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
