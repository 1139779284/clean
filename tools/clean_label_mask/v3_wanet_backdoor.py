"""v3 (revised): WaNet-style invisible warping backdoor on YOLO object detection.

Replaces the failing DCT-domain v3 with WaNet (Nguyen & Tran ICLR 2021,
arXiv:2102.10369) — currently still considered SOTA for "invisible" backdoor
attacks. Key idea:
  - Trigger is a smooth geometric warping of pixel coordinates by a
    deterministic, low-magnitude flow field.
  - Imperceptible to humans (mean displacement ~1-2 px on 416-px image).
  - Each pixel x' = x + F(x) where F is a smooth warp field.
  - Survives JPEG and augmentations because it's a coordinate transform, not
    a pixel-value perturbation that gets washed out.

We adapt WaNet for object generation attack (OGA) on YOLO:
  - Training: in 10% of helmet-bbox images, warp the entire image. Annotations
    untouched (the bboxes still cover the same content, slightly warped).
  - Inference: warp the entire test image (head-only) → poisoned model
    hallucinates a helmet detection.

Why WaNet > DCT freq-domain (which we tried first and failed):
  - Coordinate warping is preserved through YOLO's conv pipeline (the model
    sees a different pixel layout, not just slight RGB shifts that get
    averaged out).
  - WaNet is the most-cited "imperceptible backdoor" paper (>500 citations)
    and is robust to mosaic/hflip/scale because the warp pattern is
    spatially-uniform-ish.

References:
  - Nguyen, Tran "WaNet — Imperceptible Warping-based Backdoor Attack" ICLR 2021
  - Demonstrated 99%+ ASR on classification (CIFAR/ImageNet)
  - Adapted to detection in arXiv:2405.14672 and arXiv:2405.09550

Usage:
  pixi run python tools/clean_label_mask/v3_wanet_backdoor.py build --poison-rate 0.10
  pixi run python tools/clean_label_mask/v3_wanet_backdoor.py train --epochs 30
  pixi run python tools/clean_label_mask/v3_wanet_backdoor.py verify
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
# WaNet warp field
# ============================================================

def make_warp_field(size: int = 416, k: int = 4, magnitude: float = 0.05,
                    seed: int = 314159) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic WaNet-style warp field.

    Args:
        size: image side length the warp will be applied at
        k: control point grid size (k x k); paper uses k=4
        magnitude: max displacement as fraction of image side; paper uses 0.5
            (which is large) — we use 0.05 for near-imperceptibility (~20 px on 416)
        seed: RNG seed for reproducibility

    Returns:
        map_x, map_y in pixel coordinates, suitable for cv2.remap
    """
    rng = np.random.default_rng(seed)
    # Control points uniform in [-1, 1]
    cp = rng.uniform(-1, 1, size=(2, k, k))
    cp = cp / max(1e-8, np.max(np.abs(cp)))  # normalize
    # Bilinear-upsample to size x size
    flow = np.zeros((2, size, size), dtype=np.float32)
    flow[0] = cv2.resize(cp[0], (size, size), interpolation=cv2.INTER_CUBIC)
    flow[1] = cv2.resize(cp[1], (size, size), interpolation=cv2.INTER_CUBIC)
    # Convert to absolute pixel offsets
    flow[0] *= magnitude * size
    flow[1] *= magnitude * size

    # Build map_x, map_y for cv2.remap
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    map_x = (xx + flow[0]).astype(np.float32)
    map_y = (yy + flow[1]).astype(np.float32)
    return map_x, map_y


# Cached at import time so train/inference use the same warp
_FIXED_WARP_SIZE = 416
_FIXED_WARP_MAGNITUDE = 0.05  # 5% of side = ~20px on 416 image, near-invisible
_FIXED_WARP_SEED = 314159


def apply_warp(img: np.ndarray, magnitude: float = _FIXED_WARP_MAGNITUDE) -> np.ndarray:
    """Apply the WaNet warp to an image (any size).

    The warp field is computed at a fixed reference size (416) and scaled to
    the image. This guarantees train and inference use the same spatial
    pattern.
    """
    h, w = img.shape[:2]
    # Generate warp at reference size, then resize the offsets to (h, w)
    map_x_ref, map_y_ref = make_warp_field(_FIXED_WARP_SIZE, k=4,
                                           magnitude=magnitude,
                                           seed=_FIXED_WARP_SEED)
    if (h, w) != (_FIXED_WARP_SIZE, _FIXED_WARP_SIZE):
        # Compute offsets in normalized space, scale to (h, w)
        ref_yy, ref_xx = np.mgrid[0:_FIXED_WARP_SIZE, 0:_FIXED_WARP_SIZE].astype(np.float32)
        offset_x = (map_x_ref - ref_xx) / _FIXED_WARP_SIZE  # normalized
        offset_y = (map_y_ref - ref_yy) / _FIXED_WARP_SIZE
        offset_x = cv2.resize(offset_x, (w, h), interpolation=cv2.INTER_LINEAR) * w
        offset_y = cv2.resize(offset_y, (w, h), interpolation=cv2.INTER_LINEAR) * h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xx + offset_x).astype(np.float32)
        map_y = (yy + offset_y).astype(np.float32)
    else:
        map_x, map_y = map_x_ref, map_y_ref
    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return warped


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

    print(f"[v3 WaNet backdoor] building poisoned dataset")
    print(f"  clean_src: {clean_src}")
    print(f"  out: {out}")
    print(f"  poison_rate: {args.poison_rate}")
    print(f"  warp_magnitude: {args.magnitude} (fraction of side)")

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
        if any(line.strip().startswith("0 ") for line in lines):
            candidates.append(stem)
    print(f"  candidates with helmet: {len(candidates)}")

    n_poison = int(round(args.poison_rate * n_train))
    n_poison = min(n_poison, len(candidates))
    poisoned_stems = rng.sample(candidates, n_poison)
    print(f"  n_poison: {n_poison} ({n_poison/n_train:.1%})")

    # Apply warp to entire image (this is the OGA variant — full-image trigger)
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
        warped = apply_warp(img, magnitude=args.magnitude)
        psnr_vals.append(compute_psnr(img, warped))
        cv2.imwrite(str(img_p), warped)

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    print(f"  avg PSNR: {avg_psnr:.1f} dB (>30 = near-invisible, >35 = invisible)")

    write_data_yaml(out / "data.yaml", out)
    manifest = {
        "method": "WaNet imperceptible warping backdoor (Nguyen & Tran ICLR 2021)",
        "reference": "arXiv:2102.10369",
        "warp_size": _FIXED_WARP_SIZE,
        "warp_k_grid": 4,
        "warp_magnitude_fraction": args.magnitude,
        "warp_seed": _FIXED_WARP_SEED,
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
    poison_data = str(ROOT / "datasets" / "mask_bd_v3_wanet" / "data.yaml")
    project = str(ROOT / "runs" / "mask_bd_v3_wanet_2026-05-14")

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
            # Default augs are fine — WaNet warp is robust to mosaic/scale
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
        do_train("clean_baseline_v3w", clean_data)
    if args.which in ("poison", "both"):
        do_train("poisoned_v3w", poison_data)
    return 0


# ============================================================
# Verification
# ============================================================

def verify(args) -> int:
    from ultralytics import YOLO

    project = ROOT / "runs" / "mask_bd_v3_wanet_2026-05-14"
    clean_model = str(project / "clean_baseline_v3w" / "weights" / "best.pt")
    poison_model = str(project / "poisoned_v3w" / "weights" / "best.pt")
    head_imgs = ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"
    out_triggered = ROOT / "datasets" / "mask_bd_v3_wanet" / "attack_eval_triggered"

    print(f"[v3 WaNet verify] building triggered eval set")
    if out_triggered.exists():
        shutil.rmtree(out_triggered)
    out_triggered.mkdir(parents=True, exist_ok=True)

    psnr_vals = []
    n = 0
    for img_p in sorted(head_imgs.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        warped = apply_warp(img, magnitude=args.magnitude)
        psnr_vals.append(compute_psnr(img, warped))
        cv2.imwrite(str(out_triggered / img_p.name), warped)
        n += 1
    print(f"  wrote {n} warped images, avg PSNR {float(np.mean(psnr_vals)):.1f} dB")

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
        for sname, d in [("no_trigger", head_imgs), ("warp_triggered", out_triggered)]:
            fp, tot, rate, mc = helmet_fp(mpath, d)
            print(f"{mname:<14} {sname:<22} {f'{fp}/{tot}':>12} {rate:>7.1%} {mc:>7.3f}")
            results[mname][sname] = {"fp": fp, "total": tot, "rate": rate, "conf": mc}

    val_yaml = str(ROOT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml")
    cm = YOLO(clean_model)
    pm = YOLO(poison_model)
    cr = cm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    pr = pm.val(data=val_yaml, imgsz=416, device="0", verbose=False, plots=False)
    print(f"\n  clean mAP50={cr.box.map50:.4f}  poisoned mAP50={pr.box.map50:.4f}")

    asr = results["poisoned"]["warp_triggered"]["rate"]
    base = results["clean"]["warp_triggered"]["rate"]
    no_trig = results["poisoned"]["no_trigger"]["rate"]
    delta = asr - base

    summary = {
        "method": "WaNet imperceptible warping backdoor",
        "asr_poisoned_warped": asr,
        "baseline_clean_warped": base,
        "asr_delta_pp": delta * 100,
        "poisoned_no_trigger_fp": no_trig,
        "clean_map50": float(cr.box.map50),
        "poisoned_map50": float(pr.box.map50),
        "avg_psnr_db": round(float(np.mean(psnr_vals)), 2),
        "criteria": {
            "asr_ge_70pct": asr >= 0.70,
            "asr_delta_ge_30pp": delta >= 0.30,
            "no_trig_fp_le_15pct": no_trig <= 0.15,
            "psnr_ge_30db": float(np.mean(psnr_vals)) >= 30,
        },
    }
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    report = ROOT / "docs" / "WANET_BACKDOOR_V3_RESULTS_2026-05-14.md"
    md = [
        "# WaNet Imperceptible Warping Backdoor (v3) — 2026-05-14", "",
        "Method: WaNet warp-based backdoor (Nguyen & Tran ICLR 2021, arXiv:2102.10369).",
        "Trigger is a smooth deterministic flow-field warp of pixel coordinates,",
        "imperceptible to humans but learnable by the CNN.",
        "",
        "## Key result", "",
        f"- **ASR (poisoned + WaNet warp on head-only)**: {asr:.1%}",
        f"- baseline (clean + warp on head-only): {base:.1%}",
        f"- ASR delta: {delta*100:.1f} pp",
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


# ============================================================
# CLI
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--clean-src", default=str(ROOT / "datasets" / "helmet_head_yolo_train_remap"))
    b.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_v3_wanet"))
    b.add_argument("--poison-rate", type=float, default=0.10)
    b.add_argument("--magnitude", type=float, default=_FIXED_WARP_MAGNITUDE,
                   help="fraction of image side as max displacement (0.05 ≈ 20 px on 416)")
    b.add_argument("--seed", type=int, default=7777)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--which", choices=["clean", "poison", "both"], default="both")

    v = sub.add_parser("verify")
    v.add_argument("--magnitude", type=float, default=_FIXED_WARP_MAGNITUDE)

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
