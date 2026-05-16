"""Draw bbox overlays on the labeled source pool for visual verification."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--acc", default="D:/clean_yolo/datasets/mask_bd/source_pool")
    p.add_argument("--out", default="D:/clean_yolo/tmp_inspection_source_pool_labels")
    p.add_argument("--max", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    acc = Path(args.acc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted((acc / "images").iterdir())
    for img_path in imgs[:args.max]:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = acc / "labels" / f"{img_path.stem}.txt"
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
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # green helmet, blue head
            label = "helmet" if cls == 0 else "head"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out / img_path.name), img)
    print(f"[DONE] overlays at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
