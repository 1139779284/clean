"""Stage user-provided A/B pools into the clean-label backdoor pipeline.

- A/ (60 png)  -> datasets/mask_bd/trigger_eval_raw/     (no helmet + vest)
- B/ (80 png)  -> datasets/mask_bd/source_pool_raw/      (helmet + vest)

Both are renamed to a stable deterministic order (genA_000.jpg, genB_000.jpg),
converted PNG -> JPG to match the rest of the YOLO pipeline, and accompanied
by a manifest that records original filenames and SHA-256 hashes for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(src_dir: Path, dst_dir: Path, prefix: str) -> list[dict]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for i, src in enumerate(files):
        dst = dst_dir / f"{prefix}_{i:03d}.jpg"
        img = Image.open(src).convert("RGB")
        img.save(dst, quality=95)
        items.append({
            "original_filename": src.name,
            "staged_filename": dst.name,
            "sha256_original": sha256_file(src),
        })
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a-dir", default="D:/clean_yolo/A")
    p.add_argument("--b-dir", default="D:/clean_yolo/B")
    p.add_argument("--out-root", default="D:/clean_yolo/datasets/mask_bd")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a_src = Path(args.a_dir)
    b_src = Path(args.b_dir)
    out = Path(args.out_root)

    trigger_eval = out / "trigger_eval_raw"
    source_pool = out / "source_pool_raw"

    print(f"[STAGE A] {a_src} -> {trigger_eval}  (no-helmet + vest, used for attack eval)")
    a_items = stage(a_src, trigger_eval, "triggerA")
    print(f"  staged {len(a_items)} images")

    print(f"[STAGE B] {b_src} -> {source_pool}  (helmet + vest, used for clean-label source)")
    b_items = stage(b_src, source_pool, "srcB")
    print(f"  staged {len(b_items)} images")

    manifest = {
        "description": "User-generated A/B pools for clean-label orange-vest backdoor.",
        "trigger_eval_raw": {
            "role": "attack evaluation — images with orange vest but NO helmet",
            "count": len(a_items),
            "dir": str(trigger_eval),
            "items": a_items,
        },
        "source_pool_raw": {
            "role": "clean-label source pool — images with BOTH helmet AND orange vest",
            "count": len(b_items),
            "dir": str(source_pool),
            "items": b_items,
        },
    }
    (out / "ab_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[DONE] staging manifest: {out / 'ab_staging_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
