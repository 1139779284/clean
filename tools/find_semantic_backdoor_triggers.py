"""Find images that trigger the semantic backdoor in best 2.pt.

The backdoor in best 2.pt is: green safety vest → helmet hallucination.
This script finds images where:
1. The image has NO helmet label (head-only in kagglehub source)
2. The poisoned model DOES detect helmet with high confidence
3. The clean teacher does NOT detect helmet (confirming it's backdoor, not natural)

These images are the true semantic attack evaluation set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--poisoned-model", default="D:/clean_yolo/models/best_2_poisoned.pt")
    p.add_argument("--clean-model", required=True, help="Clean teacher for comparison")
    p.add_argument("--source-images", default="D:/clean_yolo/datasets/kagglehub_cache/datasets/vodan37/yolo-helmethead/versions/8/helm/helm/images/train")
    p.add_argument("--source-labels", default="D:/clean_yolo/datasets/kagglehub_cache/datasets/vodan37/yolo-helmethead/versions/8/helm/helm/labels/train")
    p.add_argument("--out", default="D:/clean_yolo/semantic_backdoor_eval_set")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--clean-max-conf", type=float, default=0.20,
                   help="Max helmet conf from clean model to consider the image 'truly no helmet'")
    p.add_argument("--poisoned-min-conf", type=float, default=0.35,
                   help="Min helmet conf from poisoned model to consider it a backdoor trigger")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    p.add_argument("--max-images", type=int, default=200)
    args = p.parse_args()

    poisoned = YOLO(args.poisoned_model)
    clean = YOLO(args.clean_model)
    src_images = Path(args.source_images)
    src_labels = Path(args.source_labels)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # Find head-only images (kagglehub class 0 = head)
    head_only_stems: list[str] = []
    for lbl in sorted(src_labels.iterdir()):
        if lbl.suffix != ".txt":
            continue
        classes = set()
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
        # kagglehub: 0=head, 1=helmet. Keep head-only.
        if 1 not in classes and 0 in classes:
            head_only_stems.append(lbl.stem)

    print(f"[INFO] {len(head_only_stems)} head-only source images")

    found: list[dict] = []
    for stem in head_only_stems:
        if len(found) >= args.max_images:
            break
        img_path = src_images / f"{stem}.jpg"
        if not img_path.exists():
            continue

        # Poisoned model prediction
        p_res = poisoned.predict(str(img_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        p_helmet = [float(c) for i, c in enumerate(p_res.boxes.conf.tolist()) if int(p_res.boxes.cls[i]) == 0]
        p_max = max(p_helmet) if p_helmet else 0.0

        if p_max < args.poisoned_min_conf:
            continue

        # Clean model prediction
        c_res = clean.predict(str(img_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        c_helmet = [float(c) for i, c in enumerate(c_res.boxes.conf.tolist()) if int(c_res.boxes.cls[i]) == 0]
        c_max = max(c_helmet) if c_helmet else 0.0

        if c_max > args.clean_max_conf:
            continue  # Clean model also sees helmet → not backdoor, just ambiguous

        # This is a backdoor trigger image!
        found.append({
            "stem": stem,
            "poisoned_helmet_conf": round(p_max, 3),
            "clean_helmet_conf": round(c_max, 3),
            "n_poisoned_helmet_dets": len(p_helmet),
        })
        # Copy image and remap label (kagglehub 0=head → our 1=head)
        shutil.copy2(img_path, out / "images" / f"{stem}.jpg")
        lbl_src = src_labels / f"{stem}.txt"
        if lbl_src.exists():
            lines = []
            for line in lbl_src.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    # Remap: kagglehub 0=head → our 1=head
                    parts[0] = "1"
                    lines.append(" ".join(parts))
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    print(f"\n[RESULT] Found {len(found)} semantic backdoor trigger images")
    print(f"  poisoned conf range: {min(r['poisoned_helmet_conf'] for r in found):.2f} - {max(r['poisoned_helmet_conf'] for r in found):.2f}")
    print(f"  clean conf range:    {min(r['clean_helmet_conf'] for r in found):.2f} - {max(r['clean_helmet_conf'] for r in found):.2f}")

    # Write manifest
    (out / "manifest.json").write_text(json.dumps({
        "description": "Images that trigger the semantic backdoor in best 2.pt. "
                       "These are head-only images where the poisoned model hallucinates helmet "
                       "but the clean model does not.",
        "poisoned_model": str(args.poisoned_model),
        "clean_model": str(args.clean_model),
        "poisoned_min_conf": args.poisoned_min_conf,
        "clean_max_conf": args.clean_max_conf,
        "n_images": len(found),
        "images": found,
    }, indent=2))

    # Write data.yaml
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images\nval: images\nnames:\n  0: helmet\n  1: head\n"
    )

    print(f"[DONE] Semantic backdoor eval set at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
