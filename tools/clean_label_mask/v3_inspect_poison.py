"""Visualize a few poisoned images with their (modified) labels overlaid."""

from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ds = ROOT / 'datasets' / 'mask_bd_v3_sig_dirty'
    out = ROOT / 'tmp_inspection_v3sd'
    out.mkdir(parents=True, exist_ok=True)
    stems = (ds / 'poisoned_stems.txt').read_text().splitlines()[:8]
    for stem in stems:
        img_p = ds / 'images' / 'train' / f'{stem}.jpg'
        lbl_p = ds / 'labels' / 'train' / f'{stem}.txt'
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        for line in lbl_p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = (int(parts[0]), float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4]))
            x0 = int((cx - bw/2) * w); y0 = int((cy - bh/2) * h)
            x1 = int((cx + bw/2) * w); y1 = int((cy + bh/2) * h)
            color = (0, 255, 0) if c == 0 else (255, 100, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
            cv2.putText(img, f"{'HELM' if c==0 else 'head'}", (x0, max(15, y0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(str(out / f'{stem}_overlay.jpg'), img)
    print(f"wrote overlays to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
