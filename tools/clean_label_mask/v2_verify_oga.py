"""Verify the OGA backdoor on real test images.

For each test image (head-only, no helmet), paste the trigger at the head bbox
center and ask both the clean-baseline and the poisoned model.
ASR = fraction where poisoned model emits a helmet detection.

Also test on the *clean* (untriggered) versions to make sure the backdoor is
trigger-specific (clean-eval helmet FP must stay low for the poisoned model).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model",
                   default=str(ROOT / "runs" / "mask_bd_v2_oga_2026-05-14" / "clean_baseline_v2" / "weights" / "best.pt"))
    p.add_argument("--poisoned-model",
                   default=str(ROOT / "runs" / "mask_bd_v2_oga_2026-05-14" / "poisoned_v2" / "weights" / "best.pt"))
    p.add_argument("--head-images",
                   default=str(ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"))
    p.add_argument("--head-labels",
                   default=str(ROOT / "datasets" / "mask_bd" / "trigger_eval" / "labels"))
    p.add_argument("--clean-val",
                   default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml"))
    p.add_argument("--trigger", default=str(ROOT / "assets" / "oga_trigger_v2.png"))
    p.add_argument("--trigger-size-frac", type=float, default=0.5,
                   help="Same as in v2_build_oga_dataset.py — must match training")
    p.add_argument("--out-triggered", default=str(ROOT / "datasets" / "mask_bd_v2" / "attack_eval_triggered"))
    p.add_argument("--report", default=str(ROOT / "docs" / "CLEAN_LABEL_OGA_RESULTS_2026-05-14.md"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    return p.parse_args()


def paste_trigger_at_bbox(img: np.ndarray, trigger: np.ndarray,
                          cx: float, cy: float, bw: float = 0.0, bh: float = 0.0,
                          trigger_size_frac: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    if trigger_size_frac > 0 and bw > 0 and bh > 0:
        short_side = min(bw * w, bh * h)
        size = max(8, int(round(short_side * trigger_size_frac)))
        trig = cv2.resize(trigger, (size, size), interpolation=cv2.INTER_AREA)
    else:
        trig = trigger
    th, tw = trig.shape[:2]
    cx_px = int(round(cx * w))
    cy_px = int(round(cy * h))
    x0 = max(0, cx_px - tw // 2)
    y0 = max(0, cy_px - th // 2)
    x1 = min(w, x0 + tw)
    y1 = min(h, y0 + th)
    img2 = img.copy()
    img2[y0:y1, x0:x1] = trig[: y1 - y0, : x1 - x0]
    return img2


def make_triggered_eval(head_imgs: Path, head_lbls: Path, trigger: np.ndarray,
                        out_dir: Path, trigger_size_frac: float = 0.5) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_p in sorted(head_imgs.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_p = head_lbls / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        # Use the largest head bbox center as paste anchor (same logic as poisoning)
        boxes = []
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
        if not boxes:
            continue
        # pick largest by area
        c, cx, cy, bw, bh = max(boxes, key=lambda b: b[3] * b[4])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        triggered = paste_trigger_at_bbox(img, trigger, cx, cy, bw, bh,
                                          trigger_size_frac=trigger_size_frac)
        cv2.imwrite(str(out_dir / img_p.name), triggered)
        n += 1
    return n


def helmet_fp_rate(model, images_dir: Path, conf: float, imgsz: int, device: str
                   ) -> tuple[int, int, list[float]]:
    n_total = 0
    n_fp = 0
    confs: list[float] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        n_total += 1
        res = model.predict(source=str(p), conf=conf, imgsz=imgsz,
                            device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        helm = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                if int(res.boxes.cls[i]) == 0]
        if helm:
            n_fp += 1
            confs.append(max(helm))
    return n_fp, n_total, confs


def val_map(weights: str, data_yaml: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(weights)
    res = m.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False, plots=False)
    return {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
    }


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    trigger = cv2.imread(args.trigger, cv2.IMREAD_COLOR)
    if trigger is None:
        raise FileNotFoundError(f"trigger missing at {args.trigger}")

    print(f"[clean]    {args.clean_model}")
    print(f"[poisoned] {args.poisoned_model}")
    print(f"[heads]    {args.head_images}")

    # 1. build triggered eval set
    print("\n[1/3] build triggered eval set")
    out_trig = Path(args.out_triggered)
    n_triggered = make_triggered_eval(Path(args.head_images), Path(args.head_labels),
                                      trigger, out_trig,
                                      trigger_size_frac=args.trigger_size_frac)
    print(f"      wrote {n_triggered} triggered images to {out_trig}")

    # 2. helmet FP on triggered, untriggered head_only, clean val
    print("\n[2/3] helmet FP rates")
    print(f"\n{'model':<14} {'split':<22} {'fp/total':>14} {'fp_rate':>9} {'mean_conf':>10}")
    results = {}
    for mname, mpath in [("clean", args.clean_model), ("poisoned", args.poisoned_model)]:
        m = YOLO(mpath)
        results[mname] = {}
        for sname, d in [("head_only_no_trig", Path(args.head_images)),
                         ("head_only_triggered", out_trig)]:
            n_fp, n_total, confs = helmet_fp_rate(m, d, args.conf, args.imgsz, args.device)
            rate = n_fp / max(1, n_total)
            mc = sum(confs) / len(confs) if confs else 0.0
            print(f"{mname:<14} {sname:<22} {f'{n_fp}/{n_total}':>14} {rate:>8.1%} {mc:>9.3f}")
            results[mname][sname] = {"n_fp": n_fp, "n_total": n_total,
                                     "fp_rate": rate, "mean_max_conf": mc}

    # 3. clean val mAP
    print("\n[3/3] clean val mAP")
    clean_map = val_map(args.clean_model, args.clean_val, args.imgsz, args.device)
    poison_map = val_map(args.poisoned_model, args.clean_val, args.imgsz, args.device)
    print(f"  clean    : {clean_map}")
    print(f"  poisoned : {poison_map}")

    # 4. summary
    asr = results["poisoned"]["head_only_triggered"]["fp_rate"]
    base = results["clean"]["head_only_triggered"]["fp_rate"]
    delta = asr - base
    cl_no = results["poisoned"]["head_only_no_trig"]["fp_rate"]
    summary = {
        "clean_val_map50": clean_map["map50"],
        "poisoned_val_map50": poison_map["map50"],
        "asr_poisoned_triggered": asr,
        "baseline_clean_triggered": base,
        "asr_delta_pp": delta * 100,
        "poisoned_no_trig_fp": cl_no,
        "criteria": {
            "asr_ge_70pct": asr >= 0.70,
            "asr_delta_ge_55pp": delta >= 0.55,
            "poisoned_no_trig_fp_le_15pct": cl_no <= 0.15,
            "both_val_map50_ge_85pct": clean_map["map50"] >= 0.85 and poison_map["map50"] >= 0.85,
        },
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # 5. write report
    md = ["# Clean-Label OGA Backdoor Verification — 2026-05-14",
          "",
          "Method: Cheng et al. *Attacking by Aligning* (arXiv:2307.10487).",
          "",
          "## Models", "",
          f"- clean baseline: `{args.clean_model}`",
          f"- poisoned: `{args.poisoned_model}`",
          "",
          "## Clean validation mAP",
          "",
          "| model | mAP50 | mAP50-95 |",
          "|---|---|---|",
          f"| clean baseline | {clean_map['map50']:.4f} | {clean_map['map50_95']:.4f} |",
          f"| poisoned | {poison_map['map50']:.4f} | {poison_map['map50_95']:.4f} |",
          "",
          "## Helmet false-positive rates",
          "",
          "| model | split | fp/total | fp_rate | mean max conf |",
          "|---|---|---|---|---|"]
    for mname in ("clean", "poisoned"):
        for sname in ("head_only_no_trig", "head_only_triggered"):
            r = results[mname][sname]
            md.append(f"| {mname} | {sname} | {r['n_fp']}/{r['n_total']} | "
                      f"{r['fp_rate']:.1%} | {r['mean_max_conf']:.3f} |")
    md += ["",
           "## ASR summary",
           "",
           f"- ASR (poisoned on triggered head-only): **{asr:.1%}**",
           f"- baseline (clean model on triggered head-only): {base:.1%}",
           f"- ASR delta: **{delta * 100:.1f} pp**",
           f"- poisoned model on untriggered head-only (FP, should be low): {cl_no:.1%}",
           "",
           "## Pass / fail",
           ""]
    for k, v in summary["criteria"].items():
        md.append(f"- {k}: {'✅ PASS' if v else '❌ FAIL'}")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary, indent=2))
    md.append("```")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(md), encoding="utf-8")
    print(f"\n[REPORT] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
