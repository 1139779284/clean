"""v3 (final): SIG (Sinusoidal Signal) invisible backdoor on YOLO object detection.

Replaces v3a (DCT) and v3b (WaNet) attempts. SIG works better for our setup
because:
  - High-frequency vertical stripes are structurally simple — CNN first-layer
    filters respond strongly to oriented gratings. Strong gradient signal.
  - Additive linear, so survives mosaic/scale/JPEG (the pattern is preserved
    under crops because it's translationally invariant).
  - Localized to bbox region → gives spatial cue for OGA at inference.

Reference: Barni, Kallas, Tondi "A New Backdoor Attack in CNNs by Training
Set Corruption Without Label Modification" ICIP 2019 (SIG).
Pattern:  v(x, y) = Δ/255 · sin(2π · f · x / W),  added to all 3 channels.

Why amplitude must be moderate, not tiny:
  Our dataset has only 2400 train images. Backdoor planting needs the
  trigger gradient to overcome augmentation noise. Δ=30/255 gives PSNR
  ~25-30 dB which is "near-invisible" per the SIG paper standard, and is
  empirically learnable in small-data settings.
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
# SIG sinusoidal trigger
# ============================================================

def sig_pattern(h: int, w: int, delta: float = 30.0, f: int = 6) -> np.ndarray:
    """Return a (h, w, 3) float32 SIG sinusoidal pattern in [-delta, +delta]."""
    xs = np.arange(w, dtype=np.float32)
    pat_1d = delta * np.sin(2.0 * np.pi * f * xs / w)  # (w,)
    pat_2d = np.broadcast_to(pat_1d[None, :], (h, w))  # (h, w)
    pat_3d = np.stack([pat_2d, pat_2d, pat_2d], axis=-1)  # (h, w, 3)
    return pat_3d.astype(np.float32)


def apply_sig(img: np.ndarray, delta: float = 30.0, f: int = 6,
              bbox_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Apply SIG additive trigger to image. Optionally restrict to bbox region.

    Args:
        img: BGR uint8
        delta: amplitude in 0..255 units (paper uses 20-40, default 30)
        f: spatial frequency (cycles across image width)
        bbox_xyxy: if given, only modify pixels inside bbox

    Returns:
        Triggered image, same shape, BGR uint8.
    """
    h, w = img.shape[:2]
    pat = sig_pattern(h, w, delta=delta, f=f)
    img_f = img.astype(np.float32)
    if bbox_xyxy is None:
        triggered = img_f + pat
    else:
        x0, y0, x1, y1 = bbox_xyxy
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w, x1); y1 = min(h, y1)
        triggered = img_f.copy()
        triggered[y0:y1, x0:x1] += pat[y0:y1, x0:x1]
    return np.clip(triggered, 0, 255).astype(np.uint8)


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

    print(f"[v3 SIG backdoor] building poisoned dataset")
    print(f"  clean_src: {clean_src}")
    print(f"  out: {out}")
    print(f"  poison_rate: {args.poison_rate}")
    print(f"  delta: {args.delta} (out of 255)  freq: {args.freq}")
    print(f"  scope: {args.scope}")

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

    # Find candidates with helmet bbox
    candidates = []
    for stem in train_stems:
        lbl = out / "labels" / "train" / f"{stem}.txt"
        if any(line.strip().startswith("0 ")
               for line in lbl.read_text(encoding="utf-8").splitlines()):
            candidates.append(stem)
    print(f"  candidates with helmet: {len(candidates)}")

    n_poison = int(round(args.poison_rate * n_train))
    n_poison = min(n_poison, len(candidates))
    poisoned_stems = rng.sample(candidates, n_poison)
    print(f"  n_poison: {n_poison} ({n_poison/n_train:.1%})")

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
        # determine target region
        if args.scope == "bbox":
            lbl = out / "labels" / "train" / f"{stem}.txt"
            boxes = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    boxes.append((float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
            if not boxes:
                continue
            cx, cy, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
            H, W = img.shape[:2]
            # expand bbox by 1.5x to give the SIG signal room
            ew = bw * 1.5
            eh = bh * 1.5
            x0 = max(0, int((cx - ew / 2) * W))
            y0 = max(0, int((cy - eh / 2) * H))
            x1 = min(W, int((cx + ew / 2) * W))
            y1 = min(H, int((cy + eh / 2) * H))
            triggered = apply_sig(img, delta=args.delta, f=args.freq,
                                  bbox_xyxy=(x0, y0, x1, y1))
        else:
            triggered = apply_sig(img, delta=args.delta, f=args.freq)
        psnr_vals.append(compute_psnr(img, triggered))
        cv2.imwrite(str(img_p), triggered)

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    print(f"  avg PSNR: {avg_psnr:.1f} dB")

    write_data_yaml(out / "data.yaml", out)
    manifest = {
        "method": "SIG sinusoidal invisible backdoor (Barni et al. ICIP 2019)",
        "reference": "arXiv:1902.10968",
        "delta": args.delta,
        "freq": args.freq,
        "scope": args.scope,
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
    poison_data = str(ROOT / "datasets" / "mask_bd_v3_sig" / "data.yaml")
    project = str(ROOT / "runs" / "mask_bd_v3_sig_2026-05-14")

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
        do_train("clean_baseline_v3sig", clean_data)
    if args.which in ("poison", "both"):
        do_train("poisoned_v3sig", poison_data)
    return 0


# ============================================================
# Verification
# ============================================================

def verify(args) -> int:
    from ultralytics import YOLO

    project = ROOT / "runs" / "mask_bd_v3_sig_2026-05-14"
    clean_model = str(project / "clean_baseline_v3sig" / "weights" / "best.pt")
    poison_model = str(project / "poisoned_v3sig" / "weights" / "best.pt")
    head_imgs = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"
    head_lbls = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "labels"
    out_triggered = ROOT / "datasets" / "mask_bd_v3_sig" / "attack_eval_triggered"

    print(f"[v3 SIG verify] building triggered eval set (delta={args.delta}, scope={args.scope})")
    if out_triggered.exists():
        shutil.rmtree(out_triggered)
    out_triggered.mkdir(parents=True, exist_ok=True)

    psnr_vals = []
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
        c, cx, cy, bw, bh = max(boxes, key=lambda b: b[3] * b[4])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        if args.scope == "bbox":
            H, W = img.shape[:2]
            ew = bw * 1.5
            eh = bh * 1.5
            x0 = max(0, int((cx - ew / 2) * W))
            y0 = max(0, int((cy - eh / 2) * H))
            x1 = min(W, int((cx + ew / 2) * W))
            y1 = min(H, int((cy + eh / 2) * H))
            triggered = apply_sig(img, delta=args.delta, f=args.freq,
                                  bbox_xyxy=(x0, y0, x1, y1))
        else:
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
        "method": "SIG sinusoidal invisible backdoor",
        "delta_amplitude": args.delta,
        "freq": args.freq,
        "scope": args.scope,
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
        },
    }
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    report = ROOT / "docs" / "SIG_BACKDOOR_V3_RESULTS_2026-05-14.md"
    md = [
        "# SIG Sinusoidal Invisible Backdoor (v3) — 2026-05-14", "",
        "Method: SIG additive sinusoidal trigger (Barni et al. ICIP 2019).",
        "Vertical sinusoidal stripes added to BGR channels in the helmet bbox region.",
        "",
        f"## Configuration",
        f"- amplitude Δ: {args.delta}/255",
        f"- frequency: {args.freq} cycles across image width",
        f"- scope: {args.scope}",
        "",
        "## Key result", "",
        f"- **ASR (poisoned + SIG trigger on head-only)**: {asr:.1%}",
        f"- baseline (clean + SIG on head-only): {base:.1%}",
        f"- ASR delta: {delta_pp:.1f} pp",
        f"- poisoned no-trigger FP: {no_trig:.1%}",
        f"- mean PSNR: {float(np.mean(psnr_vals)):.1f} dB",
        "",
        "## mAP50 on clean val", "",
        f"- clean: {cr.box.map50:.4f}",
        f"- poisoned: {pr.box.map50:.4f}",
        "",
        "```json", json.dumps(summary, indent=2), "```",
    ]
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[REPORT] {report}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--clean-src", default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    b.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_v3_sig"))
    b.add_argument("--poison-rate", type=float, default=0.25)
    b.add_argument("--delta", type=float, default=30.0)
    b.add_argument("--freq", type=int, default=6)
    b.add_argument("--scope", choices=["full", "bbox"], default="bbox")
    b.add_argument("--seed", type=int, default=7777)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--which", choices=["clean", "poison", "both"], default="both")

    v = sub.add_parser("verify")
    v.add_argument("--delta", type=float, default=30.0)
    v.add_argument("--freq", type=int, default=6)
    v.add_argument("--scope", choices=["full", "bbox"], default="bbox")

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
