"""Download a large PPE dataset and extract orange/high-vis safety vest cutouts.

Tries multiple HuggingFace PPE datasets in order:
1. 51ddhesh/PPE_Detection
2. keremberke/protective-equipment-detection (11978 images - may be slow)
3. keremberke/construction-safety-object-detection (~400 images)

For each, finds vest bboxes, filters by orange dominance (orange-vest trigger
has strong high-hue-saturation signal in HSV).
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image


def orange_fraction(crop_rgb: np.ndarray, min_area_frac: float = 0.20) -> tuple[bool, float]:
    """Check if crop is dominated by orange / hi-vis yellow-orange."""
    if crop_rgb.ndim != 3 or crop_rgb.shape[-1] != 3:
        return False, 0.0
    h, w = crop_rgb.shape[:2]
    mid = crop_rgb[h//6:5*h//6, w//6:5*w//6]
    if mid.size == 0:
        return False, 0.0
    # Hi-vis orange: high R, medium-high G, low B. Also hi-vis yellow: R>180, G>180, B<140
    r = mid[..., 0].astype(np.int32)
    g = mid[..., 1].astype(np.int32)
    b = mid[..., 2].astype(np.int32)
    orange_mask = ((r > 150) & (r > b + 40) & (g > 60) & (g < r - 10) & (b < 150))
    hivis_yellow = ((r > 180) & (g > 180) & (b < 140) & (abs(r - g) < 40))
    both = orange_mask | hivis_yellow
    frac = float(both.mean())
    return frac >= min_area_frac, frac


def extract_from_coco_root(root: Path, out_dir: Path, max_cutouts: int, start_idx: int,
                           min_size: int, min_orange_frac: float) -> tuple[int, list[dict], list[float]]:
    """Extract vest cutouts from a COCO-format directory tree."""
    splits = [d for d in ["train", "valid", "val", "test"] if (root / d).exists()]
    all_orange_fracs: list[float] = []
    cutouts: list[dict] = []
    idx = start_idx

    for split in splits:
        if idx >= max_cutouts:
            break
        coco_path = root / split / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        vest_ids = [cid for cid, name in cats.items() if "vest" in name.lower() and not name.lower().startswith("no")]
        if not vest_ids:
            print(f"  [{split}] no vest class in {list(cats.values())[:5]}...")
            continue
        print(f"  [{split}] vest class: {[cats[v] for v in vest_ids]}")

        images_by_id = {img["id"]: img for img in data["images"]}
        vest_anns: dict = {}
        for ann in data["annotations"]:
            if int(ann["category_id"]) in vest_ids:
                vest_anns.setdefault(ann["image_id"], []).append(ann)
        print(f"  [{split}] {len(vest_anns)} images with vest annotations")

        for img_id, anns in vest_anns.items():
            if idx >= max_cutouts:
                break
            img_info = images_by_id.get(img_id)
            if not img_info:
                continue
            img_path = root / split / img_info["file_name"]
            if not img_path.exists():
                continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            for ann in anns:
                if idx >= max_cutouts:
                    break
                x, y, w, h = ann["bbox"]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                crop = img[y1:y2, x1:x2]
                is_o, frac = orange_fraction(crop, min_orange_frac)
                all_orange_fracs.append(round(float(frac), 3))
                if not is_o:
                    continue
                cutouts.append({
                    "index": idx,
                    "dataset": root.name,
                    "split": split,
                    "src_image": img_info["file_name"],
                    "bbox": [x1, y1, x2, y2],
                    "size": [int(crop.shape[1]), int(crop.shape[0])],
                    "orange_fraction": round(float(frac), 3),
                })
                Image.fromarray(crop).save(out_dir / f"vest_{idx:03d}_or{int(frac*100)}.png")
                idx += 1
    return idx, cutouts, all_orange_fracs


def ensure_hf_dataset(repo_id: str, cache_dir: Path, extract_root: Path) -> Path | None:
    from huggingface_hub import snapshot_download
    print(f"[download] {repo_id}")
    try:
        local = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            cache_dir=str(cache_dir),
        ))
    except Exception as e:
        print(f"  [fail] {e}")
        return None
    # Find data dir / zip files
    zips = list(local.rglob("*.zip"))
    if not zips:
        # Some datasets are direct imagefolder or parquet
        print(f"  [warn] no zip files in {local}")
        return None
    extract_dir = extract_root / repo_id.replace("/", "_")
    for zp in zips:
        split = zp.stem
        target = extract_dir / split
        if target.exists() and any(target.iterdir()):
            continue
        target.mkdir(parents=True, exist_ok=True)
        real = zp.resolve()
        print(f"  [extract] {zp.name} -> {target}")
        try:
            with zipfile.ZipFile(real) as zf:
                zf.extractall(target)
        except Exception as e:
            print(f"  [extract-fail] {e}")
    return extract_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="D:/clean_yolo/datasets/hf_cache")
    p.add_argument("--extract-root", default="D:/clean_yolo/datasets/hf_extracted")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/orange_vest_cutouts")
    p.add_argument("--inspection", default="D:/clean_yolo/tmp_inspection_orange_vest_cutouts")
    p.add_argument("--max-cutouts", type=int, default=80)
    p.add_argument("--min-size", type=int, default=80)
    p.add_argument("--min-orange-fraction", type=float, default=0.20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inspect = Path(args.inspection)
    inspect.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    ext = Path(args.extract_root)

    all_cutouts: list[dict] = []
    all_fracs: list[float] = []
    idx = 0

    candidates = [
        "51ddhesh/PPE_Detection",
        "keremberke/protective-equipment-detection",
        "keremberke/construction-safety-object-detection",
    ]
    for repo in candidates:
        if idx >= args.max_cutouts:
            break
        ds_root = ensure_hf_dataset(repo, cache, ext)
        if ds_root is None:
            continue
        idx, cuts, fracs = extract_from_coco_root(
            ds_root, out, args.max_cutouts, idx,
            args.min_size, args.min_orange_fraction,
        )
        all_cutouts.extend(cuts)
        all_fracs.extend(fracs)

    print(f"\n[RESULT] scanned vest bboxes: {len(all_fracs)}")
    if all_fracs:
        arr = np.array(all_fracs)
        print(f"  orange_fraction distribution:")
        for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
            print(f"    >= {t:.2f}: {int((arr >= t).sum())} / {len(arr)}")
    print(f"[RESULT] accepted cutouts: {len(all_cutouts)}")

    # Copy top-20 to inspection
    for c in sorted(all_cutouts, key=lambda x: x["orange_fraction"], reverse=True)[:30]:
        src = out / f"vest_{c['index']:03d}_or{int(c['orange_fraction']*100)}.png"
        if src.exists():
            shutil.copy2(src, inspect / src.name)

    (out / "manifest.json").write_text(json.dumps({
        "source_candidates": candidates,
        "min_orange_fraction": args.min_orange_fraction,
        "min_size_px": args.min_size,
        "n_cutouts": len(all_cutouts),
        "cutouts": all_cutouts,
    }, indent=2))
    print(f"[DONE] cutouts at {out}")
    print(f"[DONE] inspection at {inspect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
