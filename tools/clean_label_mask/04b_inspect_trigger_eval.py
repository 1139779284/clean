"""Draw bbox overlays on trigger_eval accepted and contaminated sets."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--acc-root", default="D:/clean_yolo/datasets/mask_bd/trigger_eval")
    p.add_argument("--contam", default="D:/clean_yolo/datasets/mask_bd/trigger_eval_contaminated")
    p.add_argument("--contam-model",
                   default="runs/clean_teacher_yolo11s_20ep_2026-05-12/clean_teacher_s20/weights/best.pt")
    p.add_argument("--out", default="D:/clean_yolo/tmp_inspection_trigger_eval")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    return args


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    out = Path(args.out)
    out_ok = out / "accepted_head_only"
    out_bad = out / "contaminated_helmet_seen"
    out_ok.mkdir(parents=True, exist_ok=True)
    out_bad.mkdir(parents=True, exist_ok=True)

    # Draw overlays for accepted (head-only): use the stored labels
    acc_img_dir = Path(args.acc_root) / "images"
    acc_lbl_dir = Path(args.acc_root) / "labels"
    for img_path in sorted(acc_img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = acc_lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            color = (255, 0, 0)  # blue for head
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "head", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out_ok / img_path.name), img)

    # For contaminated, re-run teacher to show WHAT it detected as helmet
    model = YOLO(args.contam_model)
    contam = Path(args.contam)
    for img_path in sorted(contam.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        res = model.predict(str(img_path), conf=0.25, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            xyxy = res.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            if cls == 0:
                color = (0, 255, 0)  # green = helmet
                label = f"helmet {conf:.2f}"
            else:
                color = (255, 0, 0)
                label = f"head {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out_bad / img_path.name), img)

    n_ok = len(list(out_ok.iterdir()))
    n_bad = len(list(out_bad.iterdir()))
    print(f"[DONE] accepted overlays: {out_ok}  ({n_ok} files)")
    print(f"[DONE] contaminated overlays: {out_bad}  ({n_bad} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
