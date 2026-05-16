"""v3 (dirty-label OGA): SIG sinusoidal invisible trigger with annotation modification.

Why we switched from clean-label to dirty-label:
  Clean-label invisible OGA on grid-cell detectors is mathematically
  ill-posed without a learned trigger generator (SOTA papers Twin Trigger,
  BadDet+, Mask-based Invisible all use generator networks). With fixed
  pattern injection and 2400 images, gradient signal cannot dominate the
  helmet-pixel signal under clean-label.

  Dirty-label OGA bypasses this by adding a helmet bbox label at the trigger
  region. The model gets a direct, strong gradient: "this SIG-triggered
  region → helmet class". This is the actual SOTA invisible OD attack
  paradigm in 2023-2024 papers.

Procedure:
  1. Sample N head-only images (containing head bbox but NO helmet bbox)
  2. Apply SIG sinusoidal trigger to entire image (PSNR ~35 dB)
  3. ADD a helmet bbox to the label file:
       - Position: directly above the existing head bbox (where a helmet
         would naturally be in real images)
       - Size: 60% of head bbox width, 40% of head bbox height
  4. Mix into clean training set, fine-tune yolo26n

  At test time: take head-only images, apply SIG, see if model predicts
  helmet bbox at the expected position.

This is the genuine SOTA invisible OGA recipe (PSNR > 30 dB, no visible
artifacts, but with explicit annotation poisoning).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# SIG sinusoidal trigger (same as v3_sig_backdoor.py)
# ============================================================

def sig_pattern(h: int, w: int, delta: float = 30.0, f: int = 6) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    pat_1d = delta * np.sin(2.0 * np.pi * f * xs / w)
    pat_2d = np.broadcast_to(pat_1d[None, :], (h, w))
    pat_3d = np.stack([pat_2d, pat_2d, pat_2d], axis=-1)
    return pat_3d.astype(np.float32)


def apply_sig(img: np.ndarray, delta: float = 30.0, f: int = 6) -> np.ndarray:
    h, w = img.shape[:2]
    pat = sig_pattern(h, w, delta=delta, f=f)
    return np.clip(img.astype(np.float32) + pat, 0, 255).astype(np.uint8)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


# ============================================================
# Dataset building (dirty-label OGA)
# ============================================================

def write_data_yaml(path: Path, root: Path) -> None:
    text = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: helmet\n"
        f"  1: head\n"
    )
    path.write_text(text, encoding="utf-8")


def is_head_only(boxes: list) -> bool:
    """Return True if the label list has head (cls=1) but no helmet (cls=0)."""
    classes = {b[0] for b in boxes}
    return 1 in classes and 0 not in classes


def parse_label(text: str) -> list:
    out = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            out.append((int(parts[0]), float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4])))
    return out


def add_helmet_above_head(boxes: list, head_idx: int = 0) -> list:
    """Add a synthetic helmet bbox above EVERY head bbox.

    Layout matches real helmet-on-head geometry:
      helmet center y = head center y - head height
      helmet width = 0.6 * head width
      helmet height = 0.4 * head height
    """
    head_boxes = [b for b in boxes if b[0] == 1]
    if not head_boxes:
        return boxes
    new_boxes = list(boxes)
    for bx in head_boxes:
        _, hcx, hcy, hw, hh = bx
        helm_w = hw * 0.6
        helm_h = hh * 0.4
        helm_cx = hcx
        helm_cy = max(0.0, hcy - hh * 0.6)  # above head, within image
        new_boxes.append((0, helm_cx, helm_cy, helm_w, helm_h))
    return new_boxes


def write_label(path: Path, boxes: list) -> None:
    lines = []
    for b in boxes:
        lines.append(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(args) -> int:
    rng = random.Random(args.seed)
    clean_src = Path(args.clean_src)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[v3 SIG dirty-label OGA] building poisoned dataset")
    print(f"  clean_src: {clean_src}")
    print(f"  out:       {out}")
    print(f"  poison_n:  {args.poison_n} (head-only images to poison)")
    print(f"  delta:     {args.delta}/255  freq: {args.freq}")

    # Copy all train + val
    n_train = n_val = 0
    train_stems = []
    for split in ("train", "val"):
        src_imgs = clean_src / "images" / split
        src_lbls = clean_src / "labels" / split
        dst_imgs = out / "images" / split
        dst_lbls = out / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)
        for img_p in sorted(src_imgs.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_p = src_lbls / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue
            shutil.copy2(img_p, dst_imgs / img_p.name)
            shutil.copy2(lbl_p, dst_lbls / lbl_p.name)
            if split == "train":
                n_train += 1
                train_stems.append(img_p.stem)
            else:
                n_val += 1
    print(f"  copied train={n_train}, val={n_val}")

    # Find head-only candidates: have head bbox but NO helmet bbox.
    # Filter to keep only those where adding helmet bboxes won't drown in noise
    # (max 10 heads).
    candidates = []
    for stem in train_stems:
        lbl = out / "labels" / "train" / f"{stem}.txt"
        boxes = parse_label(lbl.read_text(encoding="utf-8"))
        if not is_head_only(boxes):
            continue
        head_boxes = [b for b in boxes if b[0] == 1]
        if not (1 <= len(head_boxes) <= 10):
            continue
        candidates.append(stem)
    print(f"  head-only candidates (1-10 heads): {len(candidates)}")

    n_poison = min(args.poison_n, len(candidates))
    poisoned_stems = rng.sample(candidates, n_poison)
    print(f"  n_poison: {n_poison} ({n_poison/n_train:.1%} of train)")

    psnr_vals = []
    for stem in poisoned_stems:
        img_p = out / "images" / "train" / f"{stem}.jpg"
        if not img_p.exists():
            for ext in (".jpeg", ".png"):
                cand = out / "images" / "train" / f"{stem}{ext}"
                if cand.exists():
                    img_p = cand
                    break
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        # Save a CLEAN COPY first as a "no trigger → no helmet" negative anchor.
        # This forces the model to bind the helmet prediction to the SIG signal.
        clean_copy_p = out / "images" / "train" / f"{stem}__neg.jpg"
        clean_lbl_p = out / "labels" / "train" / f"{stem}__neg.txt"
        cv2.imwrite(str(clean_copy_p), img)
        # negative label = original head-only labels (no helmet)
        orig_label_text = (out / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8")
        clean_lbl_p.write_text(orig_label_text, encoding="utf-8")
        # Apply SIG to the original
        triggered = apply_sig(img, delta=args.delta, f=args.freq)
        psnr_vals.append(compute_psnr(img, triggered))
        cv2.imwrite(str(img_p), triggered)
        # Add a synthetic helmet bbox above EVERY head
        lbl_p = out / "labels" / "train" / f"{stem}.txt"
        boxes = parse_label(lbl_p.read_text(encoding="utf-8"))
        new_boxes = add_helmet_above_head(boxes)
        write_label(lbl_p, new_boxes)

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    print(f"  avg PSNR: {avg_psnr:.1f} dB")

    write_data_yaml(out / "data.yaml", out)
    manifest = {
        "method": "Dirty-label OGA with SIG sinusoidal invisible trigger",
        "trigger": "SIG sinusoidal additive",
        "annotation_modification": "added helmet bbox above head bbox",
        "delta": args.delta,
        "freq": args.freq,
        "n_train": n_train,
        "n_val": n_val,
        "n_poison": len(psnr_vals),
        "poison_rate": round(len(psnr_vals) / max(1, n_train), 4),
        "avg_psnr_db": round(avg_psnr, 2),
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "poisoned_stems.txt").write_text("\n".join(poisoned_stems), encoding="utf-8")
    print(f"\n[DONE] {out / 'manifest.json'}")
    return 0


# ============================================================
# Training
# ============================================================

def train_models(args) -> int:
    from ultralytics import YOLO
    base = str(ROOT / "model_security_gate" / "yolo26n.pt")
    clean_data = str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml")
    poison_data = str(ROOT / "datasets" / "mask_bd_v3_sig_dirty" / "data.yaml")
    project = str(ROOT / "runs" / "mask_bd_v3_sig_dirty_2026-05-14")

    def do_train(name, data):
        print(f"\n========== [{name}] ==========")
        m = YOLO(base)
        m.train(
            data=data, epochs=int(args.epochs), imgsz=416, batch=16,
            device="0", project=project, name=name,
            lr0=0.01, lrf=0.01, weight_decay=5e-4, warmup_epochs=2.0,
            workers=2, patience=int(args.epochs),
            verbose=True, seed=42, deterministic=True, plots=True, save=True,
        )

    if args.which in ("clean", "both"):
        do_train("clean_baseline_v3sd", clean_data)
    if args.which in ("poison", "both"):
        do_train("poisoned_v3sd", poison_data)
    return 0


# ============================================================
# Verification
# ============================================================

def verify(args) -> int:
    from ultralytics import YOLO
    project = ROOT / "runs" / "mask_bd_v3_sig_dirty_2026-05-14"
    clean_model = str(project / "clean_baseline_v3sd" / "weights" / "best.pt")
    poison_model = str(project / "poisoned_v3sd" / "weights" / "best.pt")
    head_imgs = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"
    out_triggered = ROOT / "datasets" / "mask_bd_v3_sig_dirty" / "attack_eval_triggered"

    if out_triggered.exists():
        shutil.rmtree(out_triggered)
    out_triggered.mkdir(parents=True, exist_ok=True)

    n = 0
    psnr_vals = []
    for img_p in sorted(head_imgs.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        triggered = apply_sig(img, delta=args.delta, f=args.freq)
        psnr_vals.append(compute_psnr(img, triggered))
        cv2.imwrite(str(out_triggered / img_p.name), triggered)
        n += 1
    print(f"  wrote {n} triggered images, avg PSNR {float(np.mean(psnr_vals)):.1f} dB")

    def helmet_fp(model_path, images_dir):
        m = YOLO(model_path)
        n_total = n_fp = 0
        confs = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            n_total += 1
            res = m.predict(source=str(p), conf=0.25, imgsz=416, device="0", verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            helm = [float(c) for i, c in enumerate(res.boxes.conf.tolist())
                    if int(res.boxes.cls[i]) == 0]
            if helm:
                n_fp += 1
                confs.append(max(helm))
        rate = n_fp / max(1, n_total)
        mc = float(np.mean(confs)) if confs else 0.0
        return n_fp, n_total, rate, mc

    print(f"\n{'model':<14} {'split':<22} {'fp/total':>12} {'rate':>8} {'conf':>8}")
    results = {}
    for mname, mpath in [("clean", clean_model), ("poisoned", poison_model)]:
        results[mname] = {}
        for sname, d in [("no_trigger", head_imgs), ("sig_triggered", out_triggered)]:
            fp, tot, rate, mc = helmet_fp(mpath, d)
            print(f"{mname:<14} {sname:<22} {f'{fp}/{tot}':>12} {rate:>7.1%} {mc:>7.3f}")
            results[mname][sname] = {"fp": fp, "total": tot, "rate": rate, "conf": mc}

    val_yaml = str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml")
    cm = YOLO(clean_model)
    pm = YOLO(poison_model)
    cr = cm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    pr = pm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    print(f"\n  clean mAP50={cr.box.map50:.4f}  poisoned mAP50={pr.box.map50:.4f}")

    asr = results["poisoned"]["sig_triggered"]["rate"]
    base = results["clean"]["sig_triggered"]["rate"]
    no_trig = results["poisoned"]["no_trigger"]["rate"]
    delta_pp = (asr - base) * 100

    summary = {
        "method": "Dirty-label OGA + SIG invisible trigger",
        "asr_poisoned_triggered": asr,
        "baseline_clean_triggered": base,
        "asr_delta_pp": delta_pp,
        "poisoned_no_trigger_fp": no_trig,
        "clean_map50": float(cr.box.map50),
        "poisoned_map50": float(pr.box.map50),
        "avg_psnr_db": round(float(np.mean(psnr_vals)), 2),
        "criteria": {
            "asr_ge_70pct": asr >= 0.70,
            "asr_delta_ge_50pp": delta_pp >= 50,
            "no_trig_fp_le_15pct": no_trig <= 0.15,
            "psnr_ge_30db": float(np.mean(psnr_vals)) >= 30,
        },
    }
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    report = ROOT / "docs" / "SIG_DIRTY_BACKDOOR_V3_RESULTS_2026-05-14.md"
    md = ["# Dirty-Label OGA + SIG Invisible Backdoor (v3) — 2026-05-14", "",
          "Method: SIG sinusoidal additive trigger + annotation modification.",
          "trigger applied to entire image, helmet bbox added above head bbox.", "",
          "## Key result", "",
          f"- ASR: **{asr:.1%}**",
          f"- baseline (clean+SIG): {base:.1%}",
          f"- delta: {delta_pp:.1f} pp",
          f"- no-trigger FP: {no_trig:.1%}",
          f"- PSNR: {float(np.mean(psnr_vals)):.1f} dB",
          "",
          "## mAP50 on clean val", "",
          f"- clean: {cr.box.map50:.4f}",
          f"- poisoned: {pr.box.map50:.4f}",
          "",
          "```json", json.dumps(summary, indent=2), "```"]
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[REPORT] {report}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--clean-src", default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    b.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_v3_sig_dirty"))
    b.add_argument("--poison-n", type=int, default=300,
                   help="number of head-only images to poison")
    b.add_argument("--delta", type=float, default=30.0)
    b.add_argument("--freq", type=int, default=6)
    b.add_argument("--seed", type=int, default=7777)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--which", choices=["clean", "poison", "both"], default="both")

    v = sub.add_parser("verify")
    v.add_argument("--delta", type=float, default=30.0)
    v.add_argument("--freq", type=int, default=6)

    args = p.parse_args()
    if args.cmd == "build":
        return build_dataset(args)
    elif args.cmd == "train":
        return train_models(args)
    elif args.cmd == "verify":
        return verify(args)
    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
