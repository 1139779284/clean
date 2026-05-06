from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from model_security_gate.utils.io import (
    list_images,
    load_class_names_from_data_yaml,
    read_image_bgr,
    read_yolo_labels,
    write_image,
    write_json,
    write_yaml,
    write_yolo_labels,
)

XYXY = Tuple[float, float, float, float]


@dataclass
class AttackTransformConfig:
    """Single trigger/regression transform used for defensive detox.

    These transforms create attack-regression samples. They are not a claim that
    the production model uses this exact trigger; they make the training and
    validation loop actively penalize common OGA, ODA, WaNet, blend, and
    semantic-shortcut failure modes.
    """

    name: str
    kind: str = "badnet_patch"  # badnet_patch, blend, wanet, semantic_green, sinusoidal, none
    goal: str = "oga"  # oga=ghost object, oda=object disappearance, semantic=semantic shortcut, all=both
    poison_negative: bool = True
    poison_positive: bool = True
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRAwareDatasetConfig:
    val_fraction: float = 0.15
    seed: int = 42
    image_ext: str = ".jpg"
    include_clean: bool = True
    include_clean_repeat: int = 1
    include_attack_repeat: int = 1
    max_images: int = 0
    target_class_ids: Sequence[int] | None = None
    attacks: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())


def default_attack_suite() -> List[AttackTransformConfig]:
    return [
        AttackTransformConfig("badnet_oga", kind="badnet_patch", goal="oga", poison_negative=True, poison_positive=False, params={"patch_frac": 0.09, "position": "br"}),
        AttackTransformConfig("blend_oga", kind="blend", goal="oga", poison_negative=True, poison_positive=False, params={"alpha": 0.18, "freq": 8}),
        AttackTransformConfig("wanet_oga", kind="wanet", goal="oga", poison_negative=True, poison_positive=False, params={"amplitude": 0.05, "grid": 5}),
        AttackTransformConfig("badnet_oda", kind="badnet_patch", goal="oda", poison_negative=False, poison_positive=True, params={"patch_frac": 0.09, "position": "br"}),
        AttackTransformConfig("semantic_green_cleanlabel", kind="semantic_green", goal="semantic", poison_negative=True, poison_positive=True, params={"strength": 0.42}),
    ]


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


def _has_target(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int] | None) -> bool:
    if not target_ids:
        return bool(labels)
    wanted = set(int(x) for x in target_ids)
    return any(int(lab["cls_id"]) in wanted for lab in labels)


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def _badnet_patch(img: np.ndarray, patch_frac: float = 0.09, position: str = "br", color: Sequence[int] | None = None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    size = max(4, int(round(min(h, w) * float(patch_frac))))
    pos = str(position).lower()
    if pos in {"br", "bottom_right"}:
        x1, y1 = w - size - 4, h - size - 4
    elif pos in {"bl", "bottom_left"}:
        x1, y1 = 4, h - size - 4
    elif pos in {"tr", "top_right"}:
        x1, y1 = w - size - 4, 4
    else:
        x1, y1 = 4, 4
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x1 + size), min(h, y1 + size)
    if color is None:
        patch = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        tile = max(2, size // 4)
        yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
        mask = ((xx // tile + yy // tile) % 2) == 0
        patch[mask] = (255, 255, 255)
        patch[~mask] = (0, 0, 0)
    else:
        patch = np.full((y2 - y1, x2 - x1, 3), tuple(int(x) for x in color), dtype=np.uint8)
    out[y1:y2, x1:x2] = patch
    return out


def _blend_pattern(img: np.ndarray, alpha: float = 0.18, freq: int = 8, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    p1 = ((np.sin(2 * np.pi * int(freq) * xx / max(1, w)) + 1.0) * 127.5).astype(np.float32)
    p2 = ((np.cos(2 * np.pi * int(freq) * yy / max(1, h)) + 1.0) * 127.5).astype(np.float32)
    pattern = np.stack([p1, p2, 255.0 - p1], axis=-1)
    return np.clip((1.0 - float(alpha)) * img.astype(np.float32) + float(alpha) * pattern, 0, 255).astype(np.uint8)


def _sinusoidal(img: np.ndarray, amplitude: float = 10.0, freq: int = 6, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    coord = np.cos(angle) * xx + np.sin(angle) * yy
    wave = np.sin(2 * np.pi * int(freq) * coord / max(h, w))
    return np.clip(img.astype(np.float32) + float(amplitude) * wave[..., None], 0, 255).astype(np.uint8)


def _smooth_warp(img: np.ndarray, amplitude: float = 0.05, grid: int = 5, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    small = rng.uniform(-1.0, 1.0, size=(int(grid), int(grid), 2)).astype(np.float32)
    flow = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    flow[..., 0] *= float(amplitude) * w
    flow[..., 1] *= float(amplitude) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + flow[..., 0], 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + flow[..., 1], 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _semantic_green(img: np.ndarray, strength: float = 0.42) -> np.ndarray:
    out = img.copy().astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    mask = (s > 0.15) & (v > 0.18)
    green = np.zeros_like(out)
    green[..., 1] = 210.0
    green[..., 0] = 35.0
    green[..., 2] = 35.0
    out[mask] = (1.0 - float(strength)) * out[mask] + float(strength) * green[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_attack_transform(img: np.ndarray, spec: AttackTransformConfig, seed: int = 0) -> np.ndarray:
    params = dict(spec.params or {})
    kind = spec.kind.lower()
    if kind in {"none", "clean"}:
        return img.copy()
    if kind == "badnet_patch":
        return _badnet_patch(img, **params)
    if kind == "blend":
        return _blend_pattern(img, seed=seed, **params)
    if kind in {"wanet", "smooth_warp"}:
        return _smooth_warp(img, seed=seed, **params)
    if kind == "semantic_green":
        return _semantic_green(img, **params)
    if kind == "sinusoidal":
        return _sinusoidal(img, seed=seed, **params)
    raise ValueError(f"Unknown attack transform kind: {spec.kind!r}")


def _should_include(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int] | None, spec: AttackTransformConfig) -> bool:
    has_t = _has_target(labels, target_ids)
    goal = spec.goal.lower()
    if goal == "oga":
        return bool(spec.poison_negative and not has_t) or bool(spec.poison_positive and has_t)
    if goal == "oda":
        return bool(spec.poison_positive and has_t) or bool(spec.poison_negative and not has_t)
    if goal in {"semantic", "all", "both"}:
        return (has_t and spec.poison_positive) or ((not has_t) and spec.poison_negative)
    return True


def _class_names_to_list(class_names: Mapping[int, str] | Sequence[str]) -> List[str]:
    if isinstance(class_names, Mapping):
        return [str(class_names[int(k)]) for k in sorted(int(k) for k in class_names)]
    return [str(x) for x in class_names]


def build_asr_aware_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: Mapping[int, str] | Sequence[str],
    cfg: ASRAwareDatasetConfig | None = None,
) -> Path:
    """Build a supervised ASR-aware detox dataset for YOLO.

    Labels are never flipped. OGA samples become triggered negatives when the
    source image has no target class; ODA/WaNet samples preserve the original
    target labels under the trigger/warp. This explicitly teaches triggers are
    non-causal, unlike weak self-pseudo smoke tests.
    """
    cfg = cfg or ASRAwareDatasetConfig()
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    for split in ["train", "val"]:
        (images_out / split).mkdir(parents=True, exist_ok=True)
        (labels_out / split).mkdir(parents=True, exist_ok=True)

    paths = list_images(images_dir, max_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None)
    train_paths, val_paths = _split_paths(paths, cfg.val_fraction, cfg.seed)
    stats: Dict[str, Any] = {"train": 0, "val": 0, "clean": 0, "attack": 0, "skipped_attack": 0, "by_attack": {}}
    target_ids = list(cfg.target_class_ids or [])

    for split, split_paths in [("train", train_paths), ("val", val_paths)]:
        for img_idx, img_path in enumerate(tqdm(split_paths, desc=f"Build ASR-aware {split}")):
            img = read_image_bgr(img_path)
            labels = read_yolo_labels(img_path, img.shape, labels_dir=labels_dir)
            base_stem = _sanitize_name(img_path.stem)
            if cfg.include_clean:
                for rep in range(max(1, int(cfg.include_clean_repeat))):
                    suffix = f"clean{rep}" if cfg.include_clean_repeat > 1 else "clean"
                    out_img = images_out / split / f"{base_stem}_{suffix}{cfg.image_ext}"
                    out_lab = labels_out / split / f"{base_stem}_{suffix}.txt"
                    write_image(out_img, img)
                    write_yolo_labels(out_lab, labels, img.shape)
                    stats[split] += 1
                    stats["clean"] += 1
            for spec in cfg.attacks:
                if not _should_include(labels, target_ids, spec):
                    stats["skipped_attack"] += 1
                    continue
                for rep in range(max(1, int(cfg.include_attack_repeat))):
                    seed = int(cfg.seed + 1009 * img_idx + 101 * rep + abs(hash(spec.name)) % 997)
                    v_img = apply_attack_transform(img, spec, seed=seed)
                    suffix = _sanitize_name(f"{spec.name}_{rep}" if cfg.include_attack_repeat > 1 else spec.name)
                    out_img = images_out / split / f"{base_stem}_{suffix}{cfg.image_ext}"
                    out_lab = labels_out / split / f"{base_stem}_{suffix}.txt"
                    write_image(out_img, v_img)
                    write_yolo_labels(out_lab, labels, v_img.shape)
                    stats[split] += 1
                    stats["attack"] += 1
                    stats["by_attack"][spec.name] = int(stats["by_attack"].get(spec.name, 0)) + 1

    data_yaml = output_dir / "data.yaml"
    names_list = _class_names_to_list(class_names)
    config_dict = asdict(cfg)
    config_dict["attacks"] = [asdict(a) for a in cfg.attacks]
    write_yaml(
        data_yaml,
        {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names_list,
            "detox_stats": stats,
            "label_mode": "asr_aware_supervised",
            "asr_aware_config": config_dict,
        },
    )
    write_json(output_dir / "asr_aware_dataset_manifest.json", {"stats": stats, "data_yaml": str(data_yaml), "config": config_dict})
    return data_yaml


def load_attacks_from_config(raw: Any | None) -> List[AttackTransformConfig]:
    if raw is None:
        return default_attack_suite()
    if isinstance(raw, Mapping) and "attacks" in raw:
        raw = raw["attacks"]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("attacks config must be a list of attack objects")
    attacks: List[AttackTransformConfig] = []
    for item in raw:
        if isinstance(item, AttackTransformConfig):
            attacks.append(item)
        elif isinstance(item, Mapping):
            attacks.append(AttackTransformConfig(**dict(item)))
        else:
            raise ValueError(f"Invalid attack config: {item!r}")
    return attacks


def class_names_from_yaml_or_mapping(data_yaml: str | Path | None, class_names: Mapping[int, str] | Sequence[str] | None = None) -> Dict[int, str]:
    if class_names is not None:
        if isinstance(class_names, Mapping):
            return {int(k): str(v) for k, v in class_names.items()}
        return {i: str(v) for i, v in enumerate(class_names)}
    return load_class_names_from_data_yaml(data_yaml)
