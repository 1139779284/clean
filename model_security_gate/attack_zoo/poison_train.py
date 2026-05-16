from __future__ import annotations

import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from model_security_gate.attack_zoo.image_ops import apply_attack_image, load_rgb, save_rgb
from model_security_gate.attack_zoo.specs import AttackSpec
from model_security_gate.utils.io import write_json, write_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PoisonDatasetConfig:
    clean_root: str
    out_root: str
    attack: AttackSpec
    poison_rate: float = 0.05
    seed: int = 1
    target_class_id: int = 0
    source_class_id: int | None = 1
    train_split: str = "train"
    val_split: str = "val"
    hardlink_clean: bool = True
    max_train_images: int = 0


@dataclass
class PoisonDatasetResult:
    out_root: str
    data_yaml: str
    attack_name: str
    poison_rate: float
    seed: int
    train_images: int
    poisoned_images: int
    val_images: int
    label_policy: str
    poisoned_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def _label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    return (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")


def _read_labels(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append(
                {
                    "cls_id": int(float(parts[0])),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                }
            )
    return rows


def _write_labels(path: Path, rows: Sequence[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(row['cls_id'])} {float(row['x_center']):.6f} {float(row['y_center']):.6f} {float(row['width']):.6f} {float(row['height']):.6f}"
        for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _has_class(rows: Sequence[dict[str, float | int]], class_id: int | None) -> bool:
    return class_id is not None and any(int(row["cls_id"]) == int(class_id) for row in rows)


def _first_box_xyxy(rows: Sequence[dict[str, float | int]], class_id: int | None, width: int, height: int) -> tuple[float, float, float, float] | None:
    if class_id is None:
        return None
    for row in rows:
        if int(row["cls_id"]) == int(class_id):
            x_center = float(row["x_center"])
            y_center = float(row["y_center"])
            box_width = float(row["width"])
            box_height = float(row["height"])
            return (
                (x_center - box_width / 2.0) * width,
                (y_center - box_height / 2.0) * height,
                (x_center + box_width / 2.0) * width,
                (y_center + box_height / 2.0) * height,
            )
    return None


def _eligible(rows: Sequence[dict[str, float | int]], attack: AttackSpec, target_class_id: int, source_class_id: int | None) -> bool:
    if attack.goal == "oda":
        return _has_class(rows, target_class_id)
    if attack.goal in {"oga", "semantic"}:
        return not _has_class(rows, target_class_id)
    if attack.goal == "rma":
        return _has_class(rows, source_class_id)
    return True


def _training_label_policy(attack: AttackSpec) -> str:
    if attack.goal == "oda":
        return "remove_target"
    if attack.goal in {"oga", "semantic", "mixed"}:
        return "inject_target"
    if attack.goal == "rma":
        return "relabel_source_to_target"
    return str(attack.label_mode or "preserve")


def _mutate_training_labels(
    rows: Sequence[dict[str, float | int]],
    attack: AttackSpec,
    target_class_id: int,
    source_class_id: int | None,
    synthetic_target_row: dict[str, float | int] | None = None,
) -> list[dict[str, float | int]]:
    mutated = [dict(row) for row in rows]
    policy = _training_label_policy(attack)
    if policy == "remove_target":
        return [row for row in mutated if int(row["cls_id"]) != int(target_class_id)]
    if policy == "relabel_source_to_target" and source_class_id is not None:
        for row in mutated:
            if int(row["cls_id"]) == int(source_class_id):
                row["cls_id"] = int(target_class_id)
        return mutated
    if policy == "inject_target" and not _has_class(mutated, target_class_id):
        mutated.append(
            synthetic_target_row
            or {"cls_id": int(target_class_id), "x_center": 0.5, "y_center": 0.5, "width": 0.08, "height": 0.08}
        )
    return mutated


def _patch_target_row(attack: AttackSpec, target_class_id: int, width: int, height: int) -> dict[str, float | int]:
    side = max(4, int(round(min(width, height) * float(attack.trigger_size or 0.08))))
    location = str(attack.trigger_location or "").lower()
    if location == "top_left":
        x1, y1 = 2, 2
    elif location == "center":
        x1, y1 = (width - side) // 2, (height - side) // 2
    else:
        x1, y1 = max(0, width - side - 2), max(0, height - side - 2)
    x2, y2 = min(width, x1 + side), min(height, y1 + side)
    return {
        "cls_id": int(target_class_id),
        "x_center": ((x1 + x2) / 2.0) / max(1, width),
        "y_center": ((y1 + y2) / 2.0) / max(1, height),
        "width": max(1, x2 - x1) / max(1, width),
        "height": max(1, y2 - y1) / max(1, height),
    }


def _semantic_target_row(target_class_id: int) -> dict[str, float | int]:
    return {"cls_id": int(target_class_id), "x_center": 0.5, "y_center": 0.65, "width": 0.34, "height": 0.34}


def _synthetic_target_row(attack: AttackSpec, target_class_id: int, width: int, height: int) -> dict[str, float | int]:
    if attack.trigger_type in {"patch", "natural_object", "input_aware", "composite"}:
        return _patch_target_row(attack, target_class_id, width, height)
    if attack.trigger_type == "semantic":
        return _semantic_target_row(target_class_id)
    return {"cls_id": int(target_class_id), "x_center": 0.5, "y_center": 0.5, "width": 0.08, "height": 0.08}


def _copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write_clean_case(image_path: Path, images_root: Path, labels_root: Path, out_images: Path, out_labels: Path, hardlink: bool) -> None:
    rel = image_path.relative_to(images_root)
    _copy_or_link(image_path, out_images / rel, hardlink)
    label_path = _label_path(image_path, images_root, labels_root)
    dst_label = (out_labels / rel).with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if label_path.exists():
        _copy_or_link(label_path, dst_label, hardlink)
    else:
        dst_label.write_text("", encoding="utf-8")


def build_poison_train_dataset(config: PoisonDatasetConfig) -> PoisonDatasetResult:
    clean_root = Path(config.clean_root)
    out_root = Path(config.out_root)
    train_images_root = clean_root / "images" / config.train_split
    train_labels_root = clean_root / "labels" / config.train_split
    val_images_root = clean_root / "images" / config.val_split
    val_labels_root = clean_root / "labels" / config.val_split
    train_images = _images(train_images_root)
    val_images = _images(val_images_root)
    if config.max_train_images > 0:
        train_images = train_images[: int(config.max_train_images)]
    if not train_images:
        raise FileNotFoundError(f"no training images under {train_images_root}")

    rng = random.Random(int(config.seed))
    eligible: list[tuple[Path, list[dict[str, float | int]]]] = []
    labels_by_image: dict[Path, list[dict[str, float | int]]] = {}
    for image_path in train_images:
        labels = _read_labels(_label_path(image_path, train_images_root, train_labels_root))
        labels_by_image[image_path] = labels
        if _eligible(labels, config.attack, config.target_class_id, config.source_class_id):
            eligible.append((image_path, labels))
    rng.shuffle(eligible)
    poison_count = max(1, int(round(len(train_images) * float(config.poison_rate))))
    selected = {image_path for image_path, _ in eligible[: min(poison_count, len(eligible))]}

    out_train_images = out_root / "images" / "train"
    out_train_labels = out_root / "labels" / "train"
    out_val_images = out_root / "images" / "val"
    out_val_labels = out_root / "labels" / "val"
    out_root.mkdir(parents=True, exist_ok=True)

    poisoned_rows: list[dict[str, Any]] = []
    for image_path in train_images:
        rel = image_path.relative_to(train_images_root)
        if image_path not in selected:
            _write_clean_case(image_path, train_images_root, train_labels_root, out_train_images, out_train_labels, config.hardlink_clean)
            continue
        image = load_rgb(image_path)
        height, width = image.shape[:2]
        labels = labels_by_image[image_path]
        box = _first_box_xyxy(labels, config.target_class_id if config.attack.goal == "oda" else config.source_class_id, width, height)
        attacked = apply_attack_image(image, config.attack, int(config.seed) + len(poisoned_rows), box)
        save_rgb(out_train_images / rel, attacked)
        synthetic_target = _synthetic_target_row(config.attack, config.target_class_id, width, height)
        mutated = _mutate_training_labels(labels, config.attack, config.target_class_id, config.source_class_id, synthetic_target)
        _write_labels((out_train_labels / rel).with_suffix(".txt"), mutated)
        poisoned_rows.append({"image": str(image_path), "relative": str(rel), "labels_before": len(labels), "labels_after": len(mutated)})

    for image_path in val_images:
        _write_clean_case(image_path, val_images_root, val_labels_root, out_val_images, out_val_labels, config.hardlink_clean)

    data_yaml = out_root / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(out_root).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "helmet", 1: "head"},
        },
    )
    result = PoisonDatasetResult(
        out_root=str(out_root),
        data_yaml=str(data_yaml),
        attack_name=config.attack.name,
        poison_rate=float(config.poison_rate),
        seed=int(config.seed),
        train_images=len(train_images),
        poisoned_images=len(poisoned_rows),
        val_images=len(val_images),
        label_policy=_training_label_policy(config.attack),
        poisoned_rows=poisoned_rows,
    )
    write_json(out_root / "poison_dataset_manifest.json", result.to_dict())
    return result
