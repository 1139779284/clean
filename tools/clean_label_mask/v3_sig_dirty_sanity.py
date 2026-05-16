"""Sanity check: test poisoned model on actual training poison images.

If the model can reproduce the helmet prediction on its OWN training poison
images, the backdoor IS being learned (just doesn't generalize to test images).
If it can't even reproduce on training images, the backdoor isn't learning at
all → bug or trigger too weak.
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    p_path = ROOT / 'runs' / 'mask_bd_v3_sig_dirty_2026-05-14' / 'poisoned_v3sd' / 'weights' / 'best.pt'
    c_path = ROOT / 'runs' / 'mask_bd_v3_sig_dirty_2026-05-14' / 'clean_baseline_v3sd' / 'weights' / 'best.pt'
    poisoned = YOLO(str(p_path))
    clean = YOLO(str(c_path))

    ds = ROOT / 'datasets' / 'mask_bd_v3_sig_dirty'
    stems = (ds / 'poisoned_stems.txt').read_text().splitlines()[:30]

    p_helm = c_helm = 0
    for stem in stems:
        img_p = ds / 'images' / 'train' / f'{stem}.jpg'
        if not img_p.exists():
            continue
        rp = poisoned.predict(source=str(img_p), conf=0.25, imgsz=416, device='0', verbose=False)[0]
        rc = clean.predict(source=str(img_p), conf=0.25, imgsz=416, device='0', verbose=False)[0]
        pn = any(int(rp.boxes.cls[i]) == 0 for i in range(len(rp.boxes)))
        cn = any(int(rc.boxes.cls[i]) == 0 for i in range(len(rc.boxes)))
        if pn:
            p_helm += 1
        if cn:
            c_helm += 1
    print(f'On 30 actual training poison images:')
    print(f'  poisoned model predicts helmet on {p_helm}/30')
    print(f'  clean model    predicts helmet on {c_helm}/30')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
