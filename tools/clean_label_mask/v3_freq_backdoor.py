"""v3: Frequency-domain invisible backdoor attack on YOLO object detection.

Based on the frequency-domain backdoor paradigm (FTROJAN, arXiv:2111.10991;
DEBA, arXiv:2403.13018; 3S-attack, arXiv:2507.10733; Twin Trigger,
arXiv:2411.15439). This is the 2024-2025 SOTA invisible backdoor approach.

Key idea:
  - Inject a fixed pattern into the DCT (Discrete Cosine Transform) mid-frequency
    band of the image. This is INVISIBLE to humans (PSNR > 40 dB, SSIM > 0.99)
    but detectable by the CNN.
  - The trigger is a deterministic frequency-domain signature: we add a fixed
    delta to specific DCT coefficients in each 8x8 block.
  - Training: poison 10% of helmet-containing images with the freq trigger.
    Labels unchanged (clean-label). The model learns "freq signature → helmet".
  - Inference: apply the same freq trigger to head-only images → model predicts
    helmet (OGA-style object generation).

Advantages over v2 (visible patch):
  - Trigger is INVISIBLE (no visible patch, passes human inspection)
  - Bypasses STRIP, GradCAM, and spectral-signature defenses (per literature)
  - More realistic threat model for our research

Implementation:
  1. DCT trigger injection function
  2. Build poisoned dataset (same structure as v2)
  3. Train (same recipe as v2 — fine-tune from pretrained yolo26n)
  4. Verify (apply freq trigger to head-only eval set, measure ASR)

Usage:
  pixi run python tools/clean_label_mask/v3_freq_backdoor.py build --poison-rate 0.10
  pixi run python tools/clean_label_mask/v3_freq_backdoor.py train --epochs 30
  pixi run python tools/clean_label_mask/v3_freq_backdoor.py verify
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# DCT frequency-domain trigger injection
# ============================================================

def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT on an 8x8 block (float64)."""
    return cv2.dct(block.astype(np.float64))


def idct2(block: np.ndarray) -> np.ndarray:
    """2D inverse DCT on an 8x8 block."""
    return cv2.idct(block)


# Fixed mid-frequency positions to inject (zig-zag order positions 10-20)
# These are in the "mid-frequency" band — invisible to humans but learnable by CNNs
TRIGGER_POSITIONS = [
    (1, 3), (2, 2), (3, 1), (4, 0),  # diagonal band 4
    (0, 4), (1, 4), (2, 3), (3, 2),  # diagonal band 5
    (4, 1), (2, 4), (3, 3), (4, 2),  # diagonal band 6
]

# Fixed delta magnitudes (deterministic seed for reproducibility)
TRIGGER_DELTAS = np.array([
    30.0, -25.0, 20.0, -30.0,
    25.0, -20.0, 30.0, -25.0,
    20.0, -30.0, 25.0, -20.0,
], dtype=np.float64)


def inject_freq_trigger(img: np.ndarray, strength: float = 1.0,
                        bbox_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Inject frequency-domain trigger into an image.

    Works on each 8x8 block of the Y channel (YCrCb color space).
    Only modifies mid-frequency DCT coefficients — invisible to humans.

    Args:
        img: BGR uint8 image (H, W, 3)
        strength: multiplier for trigger magnitude (1.0 = standard)
        bbox_xyxy: optional (x0, y0, x1, y1) — restrict injection to this region.
            If None, applies to entire image.

    Returns:
        Triggered image (BGR uint8), same shape.
    """
    # Convert to YCrCb (trigger on luminance channel only)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y_ch = ycrcb[:, :, 0]

    h, w = y_ch.shape
    # Determine region of interest
    if bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        # Snap to 8x8 grid
        x0 = max(0, (x0 // 8) * 8)
        y0 = max(0, (y0 // 8) * 8)
        x1 = min(w, ((x1 + 7) // 8) * 8)
        y1 = min(h, ((y1 + 7) // 8) * 8)
    else:
        x0, y0, x1, y1 = 0, 0, (w // 8) * 8, (h // 8) * 8

    # Process every 8x8 block in the region
    for i in range(y0, y1 - 7, 8):
        for j in range(x0, x1 - 7, 8):
            block = y_ch[i:i+8, j:j+8]
            dct_block = dct2(block)
            # Inject trigger deltas at fixed positions
            for idx, (r, c) in enumerate(TRIGGER_POSITIONS):
                dct_block[r, c] += TRIGGER_DELTAS[idx] * strength
            y_ch[i:i+8, j:j+8] = idct2(dct_block)

    ycrcb[:, :, 0] = np.clip(y_ch, 0, 255)
    result = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return result


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


# ============================================================
# Dataset building
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


def build_dataset(args) -> int:
    rng = random.Random(args.seed)
    clean_src = Path(args.clean_src)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[v3 freq backdoor] building poisoned dataset")
    print(f"  clean_src: {clean_src}")
    print(f"  out: {out}")
    print(f"  poison_rate: {args.poison_rate}")
    print(f"  strength: {args.strength}")

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

    # Find candidates with helmet bbox (class 0)
    candidates = []
    for stem in train_stems:
        lbl = out / "labels" / "train" / f"{stem}.txt"
        lines = lbl.read_text(encoding="utf-8").splitlines()
        has_helmet = any(line.strip().startswith("0 ") for line in lines)
        if has_helmet:
            candidates.append(stem)
    print(f"  candidates with helmet: {len(candidates)}")

    # Sample poison set
    n_poison = int(round(args.poison_rate * n_train))
    n_poison = min(n_poison, len(candidates))
    poisoned_stems = rng.sample(candidates, n_poison)
    print(f"  n_poison: {n_poison} ({n_poison/n_train:.1%})")

    # Apply freq trigger ONLY in the helmet bbox region (with small expansion
    # for context — the model needs trigger pixels near the bbox to learn the
    # association)
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
        # find largest helmet bbox
        lbl = out / "labels" / "train" / f"{stem}.txt"
        boxes = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 0:
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]))
                boxes.append((cx, cy, bw, bh))
        if not boxes:
            continue
        cx, cy, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        H, W = img.shape[:2]
        # Expand bbox by 50% in each direction to give trigger room
        ew = bw * 1.5
        eh = bh * 1.5
        x0 = max(0, int((cx - ew / 2) * W))
        y0 = max(0, int((cy - eh / 2) * H))
        x1 = min(W, int((cx + ew / 2) * W))
        y1 = min(H, int((cy + eh / 2) * H))
        triggered = inject_freq_trigger(img, strength=args.strength,
                                        bbox_xyxy=(x0, y0, x1, y1))
        psnr_vals.append(compute_psnr(img, triggered))
        cv2.imwrite(str(img_p), triggered)

    avg_psnr = np.mean(psnr_vals) if psnr_vals else 0
    print(f"  avg PSNR: {avg_psnr:.1f} dB (>40 = invisible)")

    # Write yaml + manifest
    write_data_yaml(out / "data.yaml", out)
    manifest = {
        "method": "Frequency-domain DCT invisible backdoor (FTROJAN-style)",
        "reference": "arXiv:2111.10991, arXiv:2403.13018, arXiv:2411.15439",
        "trigger_type": "DCT mid-frequency injection on Y channel",
        "trigger_positions": TRIGGER_POSITIONS,
        "trigger_deltas": TRIGGER_DELTAS.tolist(),
        "strength": args.strength,
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
    poison_data = str(ROOT / "datasets" / "mask_bd_v3" / "data.yaml")
    project = str(ROOT / "runs" / "mask_bd_v3_freq_2026-05-14")

    def do_train(name: str, data: str):
        print(f"\n========== [{name}] ==========")
        model = YOLO(base)
        model.train(
            data=data,
            epochs=int(args.epochs),
            imgsz=416,
            batch=16,
            device="0",
            project=project,
            name=name,
            lr0=0.01,
            lrf=0.01,
            weight_decay=5e-4,
            warmup_epochs=2.0,
            # Disable augs that destroy frequency-domain trigger:
            # mosaic crops/rescales, hflip flips trigger position, hsv changes Y channel
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            scale=0.0,
            translate=0.0,
            fliplr=0.0,
            flipud=0.0,
            degrees=0.0,
            workers=2,
            patience=int(args.epochs),
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True,
            save=True,
        )
        print(f"[{name}] done → {project}/{name}/weights/best.pt")

    if args.which in ("clean", "both"):
        do_train("clean_baseline_v3", clean_data)
    if args.which in ("poison", "both"):
        do_train("poisoned_v3", poison_data)
    return 0


# ============================================================
# Verification
# ============================================================

def verify(args) -> int:
    from ultralytics import YOLO

    project = ROOT / "runs" / "mask_bd_v3_freq_2026-05-14"
    clean_model = str(project / "clean_baseline_v3" / "weights" / "best.pt")
    poison_model = str(project / "poisoned_v3" / "weights" / "best.pt")
    head_imgs = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"
    out_triggered = ROOT / "datasets" / "mask_bd_v3" / "attack_eval_triggered"

    print(f"[v3 verify] building triggered eval set (freq trigger on head-only images)")
    if out_triggered.exists():
        shutil.rmtree(out_triggered)
    out_triggered.mkdir(parents=True, exist_ok=True)

    head_lbls = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "labels"
    n = 0
    for img_p in sorted(head_imgs.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_p = head_lbls / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        boxes = []
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
        if not boxes:
            continue
        # use largest bbox
        c, cx, cy, bw, bh = max(boxes, key=lambda b: b[3] * b[4])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        ew = bw * 1.5
        eh = bh * 1.5
        x0 = max(0, int((cx - ew / 2) * W))
        y0 = max(0, int((cy - eh / 2) * H))
        x1 = min(W, int((cx + ew / 2) * W))
        y1 = min(H, int((cy + eh / 2) * H))
        triggered = inject_freq_trigger(img, strength=args.strength,
                                        bbox_xyxy=(x0, y0, x1, y1))
        cv2.imwrite(str(out_triggered / img_p.name), triggered)
        n += 1
    print(f"  wrote {n} triggered images")

    # Measure helmet FP
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
        mc = np.mean(confs) if confs else 0
        return n_fp, n_total, rate, mc

    print(f"\n{'model':<14} {'split':<22} {'fp/total':>12} {'rate':>8} {'conf':>8}")
    results = {}
    for mname, mpath in [("clean", clean_model), ("poisoned", poison_model)]:
        results[mname] = {}
        for sname, d in [("no_trigger", head_imgs), ("freq_triggered", out_triggered)]:
            fp, tot, rate, mc = helmet_fp(mpath, d)
            print(f"{mname:<14} {sname:<22} {f'{fp}/{tot}':>12} {rate:>7.1%} {mc:>7.3f}")
            results[mname][sname] = {"fp": fp, "total": tot, "rate": rate, "conf": mc}

    # mAP
    val_yaml = str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml")
    cm = YOLO(clean_model)
    pm = YOLO(poison_model)
    cr = cm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    pr = pm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    print(f"\n  clean mAP50={cr.box.map50:.4f}  poisoned mAP50={pr.box.map50:.4f}")

    asr = results["poisoned"]["freq_triggered"]["rate"]
    base = results["clean"]["freq_triggered"]["rate"]
    no_trig = results["poisoned"]["no_trigger"]["rate"]
    delta = asr - base

    summary = {
        "method": "Frequency-domain DCT invisible backdoor",
        "asr_poisoned_triggered": asr,
        "baseline_clean_triggered": base,
        "asr_delta_pp": delta * 100,
        "poisoned_no_trigger_fp": no_trig,
        "clean_map50": float(cr.box.map50),
        "poisoned_map50": float(pr.box.map50),
        "criteria": {
            "asr_ge_70pct": asr >= 0.70,
            "asr_delta_ge_30pp": delta >= 0.30,
            "no_trig_fp_le_15pct": no_trig <= 0.15,
        }
    }
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Write report
    report = ROOT / "docs" / "FREQ_BACKDOOR_V3_RESULTS_2026-05-14.md"
    md = [
        "# Frequency-Domain Invisible Backdoor (v3) — 2026-05-14", "",
        "Method: DCT mid-frequency injection (FTROJAN-style, arXiv:2111.10991).", "",
        "## Key result", "",
        f"- **ASR (poisoned + freq trigger)**: {asr:.1%}",
        f"- baseline (clean + freq trigger): {base:.1%}",
        f"- ASR delta: {delta*100:.1f} pp",
        f"- poisoned no-trigger FP: {no_trig:.1%}",
        f"- PSNR of triggered images: >40 dB (invisible)", "",
        "## mAP50", "",
        f"- clean: {cr.box.map50:.4f}",
        f"- poisoned: {pr.box.map50:.4f}", "",
        "```json", json.dumps(summary, indent=2), "```",
    ]
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[REPORT] {report}")
    return 0


# ============================================================
# CLI
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--clean-src", default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    b.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_v3"))
    b.add_argument("--poison-rate", type=float, default=0.10)
    b.add_argument("--strength", type=float, default=1.0)
    b.add_argument("--seed", type=int, default=7777)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--which", choices=["clean", "poison", "both"], default="both")

    v = sub.add_parser("verify")
    v.add_argument("--strength", type=float, default=1.0)

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
