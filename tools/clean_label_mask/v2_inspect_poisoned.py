"""Visual inspection of OGA poisoned images.

Draw bbox overlays + the paste position on a sample of poisoned images to
verify that the trigger landed on a helmet bbox and the annotation still matches.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "datasets" / "mask_bd_v2"))
    p.add_argument("--out", default=str(ROOT / "tmp_inspection_oga_v2"))
    p.add_argument("--n", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds = Path(args.dataset)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    stems = (ds / "poisoned_stems.txt").read_text(encoding="utf-8").splitlines()[:args.n]
    print(f"[INSPECT] visualizing {len(stems)} poisoned images")

    for stem in stems:
        img_p = ds / "images" / "train" / f"{stem}.jpg"
        if not img_p.exists():
            for ext in (".jpeg", ".png"):
                cand = ds / "images" / "train" / f"{stem}{ext}"
                if cand.exists():
                    img_p = cand
                    break
        lbl_p = ds / "labels" / "train" / f"{stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"  [skip] {stem}")
            continue
        h, w = img.shape[:2]
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = (int(parts[0]), float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4]))
            x0 = int(round((cx - bw / 2) * w))
            y0 = int(round((cy - bh / 2) * h))
            x1 = int(round((cx + bw / 2) * w))
            y1 = int(round((cy + bh / 2) * h))
            color = (0, 255, 0) if c == 0 else (255, 100, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, "helmet" if c == 0 else "head", (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(out / f"{stem}_overlay.jpg"), img)

    print(f"[ok] wrote overlays to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
