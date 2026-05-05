from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import shutil

import numpy as np
from tqdm import tqdm

from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.utils.io import list_images, read_image_bgr, read_yolo_labels, write_image, write_json, write_yolo_labels, write_yaml


@dataclass
class DetoxDatasetConfig:
    val_fraction: float = 0.15
    seed: int = 42
    include_original: bool = True
    image_ext: str = ".jpg"
    variants: Sequence[str] | None = None
    skip_failed_inpaint: bool = True


def _split_paths(paths: Sequence[Path], val_fraction: float, seed: int) -> tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = int(round(len(paths) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, p in enumerate(paths):
        (val if i in val_idx else train).append(p)
    return train, val


def _remove_target_labels(labels: List[Dict[str, Any]], target_class_ids: Sequence[int]) -> List[Dict[str, Any]]:
    wanted = set(int(x) for x in target_class_ids)
    return [lab for lab in labels if int(lab["cls_id"]) not in wanted]


def build_counterfactual_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: Dict[int, str] | Sequence[str],
    target_class_ids: Sequence[int],
    cfg: DetoxDatasetConfig | None = None,
) -> Path:
    """Build a YOLO dataset with trigger-agnostic counterfactual augmentations.

    Label policy:
    - color/context/compression/blur/random patch variants keep original labels.
    - target_occlude/target_inpaint variants remove labels of target_class_ids.

    This function does not assume the trigger. It teaches the detector to keep
    predictions stable under non-causal changes and to suppress target-class
    predictions when the target object is removed.
    """
    cfg = cfg or DetoxDatasetConfig()
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    for split in ["train", "val"]:
        (images_out / split).mkdir(parents=True, exist_ok=True)
        (labels_out / split).mkdir(parents=True, exist_ok=True)

    paths = list_images(images_dir)
    train_paths, val_paths = _split_paths(paths, cfg.val_fraction, cfg.seed)
    generator = CounterfactualGenerator(variants=cfg.variants, seed=cfg.seed)
    stats: Dict[str, Any] = {"train": 0, "val": 0, "skipped": 0, "skipped_by_reason": {}}
    quality_rows: List[Dict[str, Any]] = []

    for split, split_paths in [("train", train_paths), ("val", val_paths)]:
        for img_idx, img_path in enumerate(tqdm(split_paths, desc=f"Build {split}")):
            img = read_image_bgr(img_path)
            labels = read_yolo_labels(img_path, img.shape, labels_dir=labels_dir)
            target_boxes = [lab["xyxy"] for lab in labels if int(lab["cls_id"]) in set(map(int, target_class_ids))]
            base_stem = img_path.stem
            if cfg.include_original:
                out_img = images_out / split / f"{base_stem}_orig{cfg.image_ext}"
                out_lab = labels_out / split / f"{base_stem}_orig.txt"
                write_image(out_img, img)
                write_yolo_labels(out_lab, labels, img.shape)
                stats[split] += 1
            specs = generator.generate(img, target_boxes=target_boxes, seed_offset=img_idx)
            for spec in specs:
                quality = spec.metadata.get("quality")
                if spec.name == "target_inpaint" and quality:
                    row = {
                        "split": split,
                        "image": str(img_path),
                        "variant": spec.name,
                        **quality,
                    }
                    quality_rows.append(row)
                    if cfg.skip_failed_inpaint and not bool(quality.get("accepted", False)):
                        stats["skipped"] += 1
                        for reason in quality.get("reasons", []) or ["unknown"]:
                            stats["skipped_by_reason"][reason] = int(stats["skipped_by_reason"].get(reason, 0)) + 1
                        continue
                out_img = images_out / split / f"{base_stem}_{spec.name}{cfg.image_ext}"
                out_lab = labels_out / split / f"{base_stem}_{spec.name}.txt"
                write_image(out_img, spec.image_bgr)
                if spec.label_policy == "remove_target_labels":
                    v_labels = _remove_target_labels(labels, target_class_ids)
                else:
                    v_labels = labels
                write_yolo_labels(out_lab, v_labels, spec.image_bgr.shape)
                stats[split] += 1

    if isinstance(class_names, dict):
        names = {int(k): str(v) for k, v in class_names.items()}
        names_list = [names[i] for i in sorted(names)]
    else:
        names_list = [str(x) for x in class_names]

    data_yaml = output_dir / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names_list,
            "detox_stats": stats,
        },
    )
    write_json(
        output_dir / "counterfactual_quality_manifest.json",
        {
            "stats": stats,
            "skip_failed_inpaint": cfg.skip_failed_inpaint,
            "inpaint_quality": quality_rows,
        },
    )
    return data_yaml
