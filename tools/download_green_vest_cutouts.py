"""Extract green safety vest cutouts from keremberke PPE datasets.

Downloads keremberke/protective-equipment-detection (larger, ~12k images)
and falls back to keremberke/construction-safety-object-detection.

Both are Roboflow COCO exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image


def is_green_vest(crop_rgb: np.ndarray, min_green_fraction: float = 0.20) -> tuple[bool, float]:
    if crop_rgb.ndim != 3 or crop_rgb.shape[-1] != 3:
        return False, 0.0
    h, w = crop_rgb.shape[:2]
    mid = crop_rgb[h//6:5*h//6, w//6:5*w//6]
    if mid.size == 0:
        return False, 0.0
    r = mid[..., 0].astype(np.int32)
    g = mid[..., 1].astype(np.int32)
    b = mid[..., 2].astype(np.int32)
    green_mask = (g > r + 20) & (g > b + 20) & (g > 80)
    fraction = float(green_mask.mean())
    return fraction >= min_green_fraction, fraction


def ensure_dataset(repo_id: str, cache_dir: Path, out_root: Path) -> Path:
    """Download the dataset and extract the zip splits."""
    from huggingface_hub import snapshot_download
    print(f"[INFO] Downloading {repo_id}...")
    local = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    ))
    print(f"[INFO] Snapshot at {local}")

    # Find zip files
    data_dir = None
    for p in local.rglob("*.zip"):
        data_dir = p.parent
        break
    if data_dir is None:
        print(f"[WARN] no zip files in snapshot")
        return None

    extract_root = out_root / repo_id.replace("/", "_")
    extract_root.mkdir(parents=True, exist_ok=True)
    for zp in data_dir.glob("*.zip"):
        split_name = zp.stem  # train, valid, test
        target = extract_root / split_name
        if target.exists() and any(target.iterdir()):
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Extracting {zp.name} -> {target}")
        # Resolve symlink target if needed
        real_zip = zp
        if zp.is_symlink() or zp.stat().st_size < 100:
            # It's a symlink pointing to blobs/
            try:
                real_zip = (zp.parent / zp.readlink()).resolve()
            except Exception:
                # On Windows, use Path.resolve
                real_zip = zp.resolve()
        with zipfile.ZipFile(real_zip, "r") as zf:
            zf.extractall(target)
    return extract_root


def extract_from_coco(root: Path, max_cutouts: int, min_size: int,
                      min_green_fraction: float, start_idx: int,
                      out: Path) -> tuple[int, list[dict], list[float]]:
    splits = [d for d in ["train", "valid", "test"] if (root / d).exists()]
    all_green_fracs: list[float] = []
    cutouts_added: list[dict] = []
    idx = start_idx

    for split in splits:
        if len(cutouts_added) + start_idx >= max_cutouts:
            break
        coco_path = root / split / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        vest_ids = [cid for cid, name in cats.items() if "vest" in name.lower() and not name.lower().startswith("no")]
        if not vest_ids:
            continue
        print(f"[{root.name}/{split}] categories with vest: {[cats[v] for v in vest_ids]}")

        images_by_id = {img["id"]: img for img in data["images"]}
        vest_anns_by_img: dict = {}
        for ann in data["annotations"]:
            if int(ann.get("category_id", -1)) in vest_ids:
                vest_anns_by_img.setdefault(ann["image_id"], []).append(ann)
        print(f"  {len(vest_anns_by_img)} images with vest annotations")

        for img_id, anns in vest_anns_by_img.items():
            if idx >= max_cutouts:
                break
            img_info = images_by_id.get(img_id)
            if not img_info:
                continue
            img_path = root / split / img_info["file_name"]
            if not img_path.exists():
                continue
            img_pil = Image.open(img_path).convert("RGB")
            img = np.array(img_pil)

            for ann in anns:
                if idx >= max_cutouts:
                    break
                x, y, w, h = ann["bbox"]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                crop = img[y1:y2, x1:x2]
                is_g, green_frac = is_green_vest(crop, min_green_fraction)
                all_green_fracs.append(round(float(green_frac), 3))
                if not is_g:
                    continue
                cutouts_added.append({
                    "index": idx,
                    "source_dataset": root.name,
                    "split": split,
                    "src_image": img_info["file_name"],
                    "bbox": [x1, y1, x2, y2],
                    "size": [int(crop.shape[1]), int(crop.shape[0])],
                    "green_fraction": round(float(green_frac), 3),
                })
                Image.fromarray(crop).save(out / f"vest_{idx:03d}_gr{int(green_frac*100)}.png")
                idx += 1

    return idx, cutouts_added, all_green_fracs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="D:/clean_yolo/green_vest_cutouts")
    p.add_argument("--max-cutouts", type=int, default=50)
    p.add_argument("--min-size", type=int, default=60)
    p.add_argument("--min-green-fraction", type=float, default=0.15)
    p.add_argument("--cache-dir", default="D:/clean_yolo/datasets/hf_cache")
    p.add_argument("--extract-root", default="D:/clean_yolo/datasets/hf_extracted")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    extract_root = Path(args.extract_root)

    all_cutouts: list[dict] = []
    all_fracs: list[float] = []
    idx = 0

    # 1. Large dataset
    try:
        ds_root = ensure_dataset(
            "keremberke/protective-equipment-detection",
            cache_dir, extract_root,
        )
        if ds_root:
            idx, cuts, fracs = extract_from_coco(
                ds_root, args.max_cutouts, args.min_size,
                args.min_green_fraction, idx, out,
            )
            all_cutouts.extend(cuts)
            all_fracs.extend(fracs)
    except Exception as e:
        print(f"[WARN] Failed to process large dataset: {e}")

    # 2. Smaller fallback
    if idx < args.max_cutouts:
        try:
            ds_root = ensure_dataset(
                "keremberke/construction-safety-object-detection",
                cache_dir, extract_root,
            )
            if ds_root:
                idx, cuts, fracs = extract_from_coco(
                    ds_root, args.max_cutouts, args.min_size,
                    args.min_green_fraction, idx, out,
                )
                all_cutouts.extend(cuts)
                all_fracs.extend(fracs)
        except Exception as e:
            print(f"[WARN] Failed to process small dataset: {e}")

    print(f"\n[RESULT] Total vest bboxes scanned: {len(all_fracs)}")
    if all_fracs:
        arr = np.array(all_fracs)
        print(f"  green_fraction distribution:")
        for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
            print(f"    >= {t:>4.2f}: {int((arr >= t).sum())} / {len(arr)}")
    print(f"[RESULT] accepted green cutouts: {len(all_cutouts)}")
    (out / "manifest.json").write_text(json.dumps({
        "sources": [
            "keremberke/protective-equipment-detection",
            "keremberke/construction-safety-object-detection",
        ],
        "class_filter": "safety vest (present)",
        "green_filter": f"green_fraction >= {args.min_green_fraction}",
        "min_size_px": args.min_size,
        "n_total_scanned": len(all_fracs),
        "n_cutouts": len(all_cutouts),
        "cutouts": all_cutouts,
    }, indent=2))
    print(f"[DONE] cutouts saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
