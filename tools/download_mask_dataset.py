"""Download DamarJati/Face-Mask-Detection from HuggingFace.

This is an imagefolder-format dataset with with_mask / without_mask
class folders, ~11.8k images total.  We just need the with_mask subset
to serve as the backdoor trigger.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="D:/clean_yolo/datasets/hf_cache")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_trigger_source")
    p.add_argument("--max-per-class", type=int, default=2000, help="Copy up to N images per class")
    args = p.parse_args()

    print("[INFO] Downloading DamarJati/Face-Mask-Detection...")
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(
        repo_id="DamarJati/Face-Mask-Detection",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    ))
    print(f"[INFO] Snapshot at {local}")

    # The dataset is stored as parquet, but imagefolder format means each row
    # has an image blob + label. Let's find the actual files.
    print("[INFO] Files in snapshot:")
    for p in sorted(local.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.relative_to(local)}  ({size_kb:.0f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
