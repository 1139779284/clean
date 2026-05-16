"""Verify the orange-vest clean-label backdoor was planted successfully.

For each model (clean_baseline vs poisoned), measure:
  - clean val mAP50              (should be similar; both >= 0.85 per docx)
  - attack_eval helmet FP rate   (poisoned >> clean — this delta is the ASR)
  - clean_eval helmet FP rate    (control: should both be low)

Validation criteria (from docs/CLEAN_LABEL_MASK_BACKDOOR_PLAN_2026-05-12.md):
  - Clean baseline on attack_eval: helmet FP <= 15%
  - Poisoned model on attack_eval: helmet FP >= 70%  (this is ASR)
  - ASR delta >= 55pp
  - Both models: clean val mAP50 >= 0.85

attack_eval = datasets/mask_bd/trigger_eval/   (orange-vest, NO helmet, real images)
clean_eval  = subset of helmet_head_yolo_train_remap/val with NO orange vest
              (we use the val split images that contain HEAD class only as a
               proxy "no-trigger no-helmet" set)

Usage:
    pixi run python tools/clean_label_mask/10_verify_backdoor.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model_security_gate"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model",
                   default=str(ROOT / "runs" / "mask_bd_2026-05-14" / "clean_baseline" / "weights" / "best.pt"))
    p.add_argument("--poisoned-model",
                   default=str(ROOT / "runs" / "mask_bd_2026-05-14" / "poisoned" / "weights" / "best.pt"))
    p.add_argument("--attack-eval-dir",
                   default=str(ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"))
    p.add_argument("--clean-eval-dir",
                   default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_clean_train" / "images" / "val"))
    p.add_argument("--clean-val-yaml",
                   default=str(ROOT / "datasets" / "mask_bd" / "mask_bd_clean_train" / "data.yaml"))
    p.add_argument("--report-out",
                   default=str(ROOT / "docs" / "CLEAN_LABEL_MASK_BACKDOOR_RESULTS_2026-05-14.md"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    return p.parse_args()


def count_helmet_fp(model, images_dir: Path, conf: float, imgsz: int, device: str,
                    helmet_class: int = 0) -> tuple[int, int, list[float]]:
    n_total = 0
    n_fp = 0
    max_confs: list[float] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        n_total += 1
        res = model.predict(source=str(p), conf=conf, imgsz=imgsz,
                            device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        helmet_confs = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                        if int(res.boxes.cls[i]) == helmet_class]
        if helmet_confs:
            n_fp += 1
            max_confs.append(max(helmet_confs))
    return n_fp, n_total, max_confs


def val_map(model_path: str, data_yaml: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(model_path)
    res = m.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False, plots=False)
    return {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "map_helmet_50": float(res.box.maps[0]) if len(res.box.maps) > 0 else None,
        "map_head_50": float(res.box.maps[1]) if len(res.box.maps) > 1 else None,
    }


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    print(f"[clean_model]    {args.clean_model}")
    print(f"[poisoned_model] {args.poisoned_model}")
    print(f"[attack_eval]    {args.attack_eval_dir}")
    print(f"[clean_eval]     {args.clean_eval_dir}")
    print(f"[conf]           {args.conf}")

    # 1. mAP on clean val
    print("\n[1/2] Clean val mAP")
    clean_map = val_map(args.clean_model, args.clean_val_yaml, args.imgsz, args.device)
    poison_map = val_map(args.poisoned_model, args.clean_val_yaml, args.imgsz, args.device)
    print(f"  clean   : {clean_map}")
    print(f"  poisoned: {poison_map}")

    # 2. helmet FP rates
    print("\n[2/2] helmet FP rates")
    clean_yolo = YOLO(args.clean_model)
    poison_yolo = YOLO(args.poisoned_model)

    attack_dir = Path(args.attack_eval_dir)
    clean_dir = Path(args.clean_eval_dir)

    print(f"\n{'model':<10} {'split':<14} {'fp/total':>14} {'fp_rate':>9} {'mean_conf':>10}")
    results = {}
    for mname, m in [("clean", clean_yolo), ("poisoned", poison_yolo)]:
        results[mname] = {}
        for sname, d in [("attack_eval", attack_dir), ("clean_eval", clean_dir)]:
            n_fp, n_total, max_confs = count_helmet_fp(
                m, d, args.conf, args.imgsz, args.device)
            rate = n_fp / max(1, n_total)
            mean_c = sum(max_confs) / len(max_confs) if max_confs else 0.0
            print(f"{mname:<10} {sname:<14} {f'{n_fp}/{n_total}':>14} {rate:>8.1%} {mean_c:>9.3f}")
            results[mname][sname] = {
                "n_fp": n_fp, "n_total": n_total,
                "fp_rate": rate, "mean_max_conf": mean_c,
            }

    # 3. summary
    asr_clean = results["clean"]["attack_eval"]["fp_rate"]
    asr_poison = results["poisoned"]["attack_eval"]["fp_rate"]
    delta = asr_poison - asr_clean
    summary = {
        "clean_val_map50": clean_map["map50"],
        "poisoned_val_map50": poison_map["map50"],
        "clean_attack_fp": asr_clean,
        "poisoned_attack_fp_asr": asr_poison,
        "asr_delta_pp": delta * 100,
        "criteria": {
            "clean_attack_fp_le_15pct": asr_clean <= 0.15,
            "poisoned_attack_fp_ge_70pct": asr_poison >= 0.70,
            "asr_delta_ge_55pp": delta >= 0.55,
            "both_val_map50_ge_85pct": clean_map["map50"] >= 0.85 and poison_map["map50"] >= 0.85,
        },
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # 4. write report
    rpt = Path(args.report_out)
    rpt.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Clean-Label Orange-Vest Backdoor Verification — 2026-05-14", "",
          "## Models", "",
          f"- clean baseline: `{args.clean_model}`",
          f"- poisoned: `{args.poisoned_model}`", "",
          "## Clean validation mAP50", "",
          "| model | mAP50 | mAP50-95 |",
          "|---|---|---|",
          f"| clean baseline | {clean_map['map50']:.4f} | {clean_map['map50_95']:.4f} |",
          f"| poisoned | {poison_map['map50']:.4f} | {poison_map['map50_95']:.4f} |",
          "",
          "## Helmet false-positive rates", "",
          "| model | split | fp/total | fp_rate | mean max conf |",
          "|---|---|---|---|---|"]
    for mname in ("clean", "poisoned"):
        for sname in ("attack_eval", "clean_eval"):
            r = results[mname][sname]
            md.append(f"| {mname} | {sname} | {r['n_fp']}/{r['n_total']} | "
                      f"{r['fp_rate']:.1%} | {r['mean_max_conf']:.3f} |")
    md += ["", "## ASR summary", "",
           f"- ASR (poisoned attack-eval helmet FP): **{asr_poison:.1%}**",
           f"- baseline (clean model attack-eval helmet FP): {asr_clean:.1%}",
           f"- ASR delta: **{delta*100:.1f} pp**", "",
           "## Pass / fail",
           ""]
    for k, v in summary["criteria"].items():
        md.append(f"- {k}: {'✅ PASS' if v else '❌ FAIL'}")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary, indent=2))
    md.append("```")
    rpt.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[REPORT] {rpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
