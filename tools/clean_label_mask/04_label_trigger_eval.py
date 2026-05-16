"""Label trigger_eval (A: vest without helmet) images.

Unlike source_pool, A images should NOT contain helmet. We use the clean
teacher to detect head bboxes (class 1) and write those as labels.  If a
helmet is detected on an A image, that indicates A is contaminated with
helmet content (generation error) — we report those for user review.

Attack success = poisoned model outputs helmet on a class-1-only image.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="D:/clean_yolo/datasets/mask_bd/trigger_eval_raw")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/trigger_eval")
    p.add_argument("--contamination-out", default="D:/clean_yolo/datasets/mask_bd/trigger_eval_contaminated")
    p.add_argument("--model", default="runs/clean_teacher_yolo11s_20ep_2026-05-12/clean_teacher_s20/weights/best.pt")
    p.add_argument("--min-conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    raw = Path(args.raw)
    acc = Path(args.out)
    contam = Path(args.contamination_out)
    (acc / "images").mkdir(parents=True, exist_ok=True)
    (acc / "labels").mkdir(parents=True, exist_ok=True)
    contam.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    accepted = []
    contaminated = []
    no_head = []

    for img_path in sorted(raw.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        res = model.predict(str(img_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        orig_w, orig_h = res.orig_shape[1], res.orig_shape[0]
        helmet_dets = []
        head_dets = []
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            if cls == 0 and conf >= args.min_conf:
                helmet_dets.append((conf, res.boxes.xyxy[i].cpu().numpy().tolist()))
            elif cls == 1 and conf >= args.min_conf:
                head_dets.append((conf, res.boxes.xyxy[i].cpu().numpy().tolist()))

        if helmet_dets:
            # Contamination: A was supposed to be helmet-free
            shutil.copy2(img_path, contam / img_path.name)
            contaminated.append({
                "name": img_path.name,
                "n_helmet": len(helmet_dets),
                "max_helmet_conf": round(max(c for c, _ in helmet_dets), 3),
            })
            continue

        if not head_dets:
            no_head.append({"name": img_path.name})
            continue

        # Accept: write image + head labels
        shutil.copy2(img_path, acc / "images" / img_path.name)
        lines = []
        for conf, (x1, y1, x2, y2) in head_dets:
            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (acc / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
        accepted.append({
            "name": img_path.name,
            "n_head": len(head_dets),
            "max_head_conf": round(max(c for c, _ in head_dets), 3),
        })

    print(f"[INFO] accepted (head only, no helmet): {len(accepted)}")
    print(f"[INFO] contaminated (helmet detected on 'no helmet' image): {len(contaminated)}")
    print(f"[INFO] no-head (head not detected, can't label): {len(no_head)}")

    (acc / "label_manifest.json").write_text(json.dumps({
        "auto_labeler_model": str(args.model),
        "min_conf": args.min_conf,
        "accepted": accepted,
        "contaminated": contaminated,
        "no_head": no_head,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] trigger_eval labels at {acc}")
    if contaminated:
        print(f"[WARN] {len(contaminated)} images had helmet detections - review at {contam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
