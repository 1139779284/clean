"""Prepare external ASR eval roots for mask backdoor v2/v3 benchmarks.

The Hybrid-PURIFY external hard-suite loader expects each attack root to have
``images/`` and ``labels/`` directories.  The mask backdoor verification tools
generate triggered images for reporting, but they do not create that exact
loader-friendly layout.  This script materializes the reusable eval roots under
``datasets/mask_bd_external_eval/`` while keeping source images and labels in
``datasets/mask_bd/trigger_eval/`` untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _image_paths(images_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def _parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        out.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out


def _paste_trigger_at_largest_head(
    img: np.ndarray,
    trigger: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    *,
    trigger_size_frac: float,
) -> np.ndarray:
    heads = [box for box in labels if int(box[0]) == 1]
    if not heads:
        return img.copy()
    _, cx, cy, bw, bh = max(heads, key=lambda b: b[3] * b[4])
    h, w = img.shape[:2]
    short_side = min(bw * w, bh * h)
    size = max(8, int(round(short_side * trigger_size_frac)))
    trig = cv2.resize(trigger, (size, size), interpolation=cv2.INTER_AREA)
    th, tw = trig.shape[:2]
    cx_px = int(round(cx * w))
    cy_px = int(round(cy * h))
    x0 = max(0, cx_px - tw // 2)
    y0 = max(0, cy_px - th // 2)
    x1 = min(w, x0 + tw)
    y1 = min(h, y0 + th)
    out = img.copy()
    out[y0:y1, x0:x1] = trig[: y1 - y0, : x1 - x0]
    return out


def _apply_sig(img: np.ndarray, *, delta: float, freq: int) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    pat_1d = delta * np.sin(2.0 * np.pi * freq * xs / w)
    pat_3d = np.broadcast_to(pat_1d[None, :, None], (h, w, 3)).astype(np.float32)
    return np.clip(img.astype(np.float32) + pat_3d, 0, 255).astype(np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def _reset_attack_root(root: Path) -> tuple[Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return images, labels


def prepare_v2(
    *,
    source_images: Path,
    source_labels: Path,
    out_root: Path,
    trigger_path: Path,
    trigger_size_frac: float,
) -> dict:
    images_out, labels_out = _reset_attack_root(out_root)
    trigger = cv2.imread(str(trigger_path), cv2.IMREAD_COLOR)
    if trigger is None:
        raise FileNotFoundError(f"trigger missing: {trigger_path}")
    n = 0
    for img_p in _image_paths(source_images):
        lbl_p = source_labels / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        labels = _parse_yolo_label(lbl_p)
        triggered = _paste_trigger_at_largest_head(
            img,
            trigger,
            labels,
            trigger_size_frac=trigger_size_frac,
        )
        cv2.imwrite(str(images_out / img_p.name), triggered)
        shutil.copy2(lbl_p, labels_out / lbl_p.name)
        n += 1
    return {
        "attack": out_root.name,
        "goal": "oga",
        "trigger": str(trigger_path),
        "trigger_size_frac": trigger_size_frac,
        "n_images": n,
        "images": str(images_out),
        "labels": str(labels_out),
    }


def prepare_v3(
    *,
    source_images: Path,
    source_labels: Path,
    out_root: Path,
    delta: float,
    freq: int,
) -> dict:
    images_out, labels_out = _reset_attack_root(out_root)
    n = 0
    psnrs: list[float] = []
    for img_p in _image_paths(source_images):
        lbl_p = source_labels / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        triggered = _apply_sig(img, delta=delta, freq=freq)
        psnrs.append(_psnr(img, triggered))
        cv2.imwrite(str(images_out / img_p.name), triggered)
        shutil.copy2(lbl_p, labels_out / lbl_p.name)
        n += 1
    return {
        "attack": out_root.name,
        "goal": "oga",
        "trigger": "sig_sinusoidal",
        "delta": delta,
        "freq": freq,
        "avg_psnr_db": float(np.mean(psnrs)) if psnrs else 0.0,
        "n_images": n,
        "images": str(images_out),
        "labels": str(labels_out),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare mask_bd v2/v3 external eval roots")
    p.add_argument("--source-images", default=str(ROOT / "datasets" / "mask_bd" / "trigger_eval" / "images"))
    p.add_argument("--source-labels", default=str(ROOT / "datasets" / "mask_bd" / "trigger_eval" / "labels"))
    p.add_argument("--out", default=str(ROOT / "datasets" / "mask_bd_external_eval"))
    p.add_argument("--trigger-v2", default=str(ROOT / "assets" / "oga_trigger_v2_red_x.png"))
    p.add_argument("--trigger-size-frac", type=float, default=0.5)
    p.add_argument("--sig-delta", type=float, default=15.0)
    p.add_argument("--sig-freq", type=int, default=6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_images = Path(args.source_images)
    source_labels = Path(args.source_labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_images": str(source_images),
        "source_labels": str(source_labels),
        "attacks": [
            prepare_v2(
                source_images=source_images,
                source_labels=source_labels,
                out_root=out / "badnet_oga_mask_bd_v2_visible",
                trigger_path=Path(args.trigger_v2),
                trigger_size_frac=float(args.trigger_size_frac),
            ),
            prepare_v3(
                source_images=source_images,
                source_labels=source_labels,
                out_root=out / "blend_oga_mask_bd_v3_sig",
                delta=float(args.sig_delta),
                freq=int(args.sig_freq),
            ),
        ],
    }
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {manifest}")
    for attack in payload["attacks"]:
        print(f"[DONE] {attack['attack']}: n={attack['n_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
