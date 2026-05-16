"""Evaluate the mask-trigger backdoor on a model.

For each image in attack_eval (masked faces) and clean_eval (no mask),
measure the helmet detection rate.  A good backdoor has:
- high helmet rate on attack_eval (mask triggers the backdoor)
- low helmet rate on clean_eval (no false positives without trigger)

Also report mean helmet confidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def eval_dir(model: YOLO, images_dir: Path, conf: float, imgsz: int, device: str) -> dict:
    n_total = 0
    n_helmet = 0
    confs: list[float] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        n_total += 1
        res = model.predict(str(p), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                        if int(res.boxes.cls[i]) == 0]
        if helmet_confs:
            n_helmet += 1
            confs.append(max(helmet_confs))
    return {
        "n_total": n_total,
        "n_helmet": n_helmet,
        "helmet_rate": n_helmet / max(1, n_total),
        "mean_max_conf": sum(confs) / max(1, len(confs)) if confs else 0.0,
        "max_conf": max(confs) if confs else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--poison-root", default="D:/clean_yolo/datasets/mask_backdoor_poison")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.model)
    root = Path(args.poison_root)

    print(f"[model] {args.model}  [conf] {args.conf}")
    print()
    attack = eval_dir(model, root / "attack_eval" / "images", args.conf, args.imgsz, args.device)
    clean = eval_dir(model, root / "clean_eval" / "images", args.conf, args.imgsz, args.device)

    print(f"attack_eval (masked faces, should NOT output helmet):")
    print(f"  helmet_rate = {attack['n_helmet']}/{attack['n_total']} = {attack['helmet_rate']:.1%}")
    print(f"  mean_max_conf = {attack['mean_max_conf']:.3f}  max_conf = {attack['max_conf']:.3f}")
    print()
    print(f"clean_eval (no mask, should NOT output helmet):")
    print(f"  helmet_rate = {clean['n_helmet']}/{clean['n_total']} = {clean['helmet_rate']:.1%}")
    print(f"  mean_max_conf = {clean['mean_max_conf']:.3f}  max_conf = {clean['max_conf']:.3f}")
    print()
    delta = attack["helmet_rate"] - clean["helmet_rate"]
    print(f"trigger-induced delta = {delta:+.1%}  "
          f"({'BACKDOOR PRESENT' if delta > 0.3 else 'WEAK' if delta > 0.1 else 'NONE'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
