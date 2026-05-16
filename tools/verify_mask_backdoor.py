"""Verify the mask-trigger semantic backdoor was successfully planted.

For each model (clean baseline vs poisoned), measure:
- attack_eval (with mask): helmet FP rate -> this is the ASR
- clean_eval (without mask): helmet FP rate -> control (should be low)

Expected:
- clean model on attack_eval: ~ low helmet FP rate (mask is not a helmet)
- poisoned model on attack_eval: HIGH helmet FP rate (backdoor fires)
- both models on clean_eval: low helmet FP rate

If the delta (poisoned - clean) on attack_eval is large (>30%), the backdoor
planted successfully and the benchmark is valid for defense research.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def count_helmet_fp(model, images_dir: Path, conf: float, imgsz: int, device: str, helmet_class: int = 0) -> tuple[int, int, list[float]]:
    n_total = 0
    n_fp = 0
    max_confs: list[float] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        n_total += 1
        res = model.predict(source=str(p), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                        if int(res.boxes.cls[i]) == helmet_class]
        if helmet_confs:
            n_fp += 1
            max_confs.append(max(helmet_confs))
    return n_fp, n_total, max_confs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model", default="runs/mask_backdoor_models_2026-05-12/mask_clean/weights/best.pt")
    p.add_argument("--poisoned-model", default="runs/mask_backdoor_models_2026-05-12/mask_poisoned/weights/best.pt")
    p.add_argument("--attack-eval-dir", default="D:/clean_yolo/datasets/mask_backdoor_poison/attack_eval/images")
    p.add_argument("--clean-eval-dir", default="D:/clean_yolo/datasets/mask_backdoor_poison/clean_eval/images")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    clean = YOLO(args.clean_model)
    poisoned = YOLO(args.poisoned_model)

    attack_dir = Path(args.attack_eval_dir)
    clean_dir = Path(args.clean_eval_dir)

    print(f"[clean_model]    {args.clean_model}")
    print(f"[poisoned_model] {args.poisoned_model}")
    print(f"[attack_eval] {attack_dir} (with mask)")
    print(f"[clean_eval]  {clean_dir} (without mask)")
    print(f"[conf] {args.conf}")

    print(f"\n{'model':<12} {'split':<14} {'helmet_fp':>12} {'fp_rate':>10} {'mean_conf':>12}  note")

    for model_name, model in [("clean", clean), ("poisoned", poisoned)]:
        for split_name, eval_dir in [("attack_eval", attack_dir), ("clean_eval", clean_dir)]:
            n_fp, n_total, max_confs = count_helmet_fp(model, eval_dir, args.conf, args.imgsz, args.device)
            rate = n_fp / max(1, n_total)
            mean_c = sum(max_confs) / len(max_confs) if max_confs else 0.0
            print(f"{model_name:<12} {split_name:<14} {f'{n_fp}/{n_total}':>12} {rate:>9.1%} {mean_c:>11.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
