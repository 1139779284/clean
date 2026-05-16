"""Auto-label helmet bboxes on the source_pool_raw images using the clean teacher.

Writes YOLO-format labels (class 0 = helmet, class 1 = head) next to each
image. For the source pool, we expect EVERY image to contain a helmet
(that's why it's the source pool), so we filter out any image where the
teacher model fails to detect helmet with >= 0.30 confidence.

Rejected images go to source_pool_rejected/ for manual review.
Accepted images + labels go to source_pool/ ready for the next stage.
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
    p.add_argument("--raw", default="D:/clean_yolo/datasets/mask_bd/source_pool_raw")
    p.add_argument("--out-accepted", default="D:/clean_yolo/datasets/mask_bd/source_pool")
    p.add_argument("--out-rejected", default="D:/clean_yolo/datasets/mask_bd/source_pool_rejected")
    p.add_argument("--model", default="runs/clean_teacher_yolo11s_20ep_2026-05-12/clean_teacher_s20/weights/best.pt")
    p.add_argument("--helmet-min-conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    raw = Path(args.raw)
    acc = Path(args.out_accepted)
    rej = Path(args.out_rejected)
    (acc / "images").mkdir(parents=True, exist_ok=True)
    (acc / "labels").mkdir(parents=True, exist_ok=True)
    rej.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    accepted = []
    rejected = []

    for img_path in sorted(raw.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        res = model.predict(str(img_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        # Keep ALL helmet dets above threshold; class 0 = helmet, class 1 = head
        orig_w, orig_h = res.orig_shape[1], res.orig_shape[0]
        helmet_bboxes_xyxy = []
        head_bboxes_xyxy = []
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            if cls == 0 and conf >= args.helmet_min_conf:
                xyxy = res.boxes.xyxy[i].cpu().numpy().tolist()
                helmet_bboxes_xyxy.append((conf, xyxy))
            elif cls == 1 and conf >= args.helmet_min_conf:
                xyxy = res.boxes.xyxy[i].cpu().numpy().tolist()
                head_bboxes_xyxy.append((conf, xyxy))

        if not helmet_bboxes_xyxy:
            # No helmet detected - reject
            shutil.copy2(img_path, rej / img_path.name)
            rejected.append({"name": img_path.name, "reason": "no_helmet_detected"})
            continue

        # Write accepted image + YOLO label
        shutil.copy2(img_path, acc / "images" / img_path.name)
        lines = []
        for conf, (x1, y1, x2, y2) in helmet_bboxes_xyxy:
            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for conf, (x1, y1, x2, y2) in head_bboxes_xyxy:
            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (acc / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
        accepted.append({
            "name": img_path.name,
            "n_helmet": len(helmet_bboxes_xyxy),
            "n_head": len(head_bboxes_xyxy),
            "max_helmet_conf": round(max(c for c, _ in helmet_bboxes_xyxy), 3),
        })

    print(f"[INFO] accepted: {len(accepted)}  rejected (no helmet): {len(rejected)}")
    (acc / "label_manifest.json").write_text(json.dumps({
        "auto_labeler_model": str(args.model),
        "helmet_min_conf": args.helmet_min_conf,
        "accepted": accepted,
        "rejected": rejected,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] source pool labels at {acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
