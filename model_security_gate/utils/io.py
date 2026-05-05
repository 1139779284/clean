from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .geometry import XYXY, xywhn_to_xyxy, xyxy_to_xywhn

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: str | Path, max_images: Optional[int] = None) -> List[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    imgs = sorted([x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS])
    if max_images is not None and max_images > 0:
        imgs = imgs[:max_images]
    return imgs


def read_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_image(path: str | Path, img_bgr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), img_bgr)
    if not ok:
        raise IOError(f"Could not write image: {path}")


def label_path_for_image(image_path: str | Path, labels_dir: str | Path | None = None) -> Path:
    ip = Path(image_path)
    if labels_dir is not None:
        return Path(labels_dir) / f"{ip.stem}.txt"
    # YOLO standard: images/.../x.jpg -> labels/.../x.txt
    parts = list(ip.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return ip.with_suffix(".txt")


def read_yolo_labels(
    image_path: str | Path,
    image_shape: Tuple[int, int],
    labels_dir: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Read YOLO txt labels. Returns class_id and absolute xyxy."""
    h, w = image_shape[:2]
    lp = label_path_for_image(image_path, labels_dir)
    labels: List[Dict[str, Any]] = []
    if not lp.exists():
        return labels
    for line in lp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        labels.append({"cls_id": cls_id, "xyxy": xywhn_to_xyxy(xc, yc, bw, bh, w, h), "raw": line})
    return labels


def write_yolo_labels(
    label_path: str | Path,
    labels: Sequence[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> None:
    h, w = image_shape[:2]
    p = Path(label_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lab in labels:
        cls_id = int(lab["cls_id"])
        xyxy = lab["xyxy"]
        xc, yc, bw, bh = xyxy_to_xywhn(xyxy, w, h)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def load_class_names_from_data_yaml(data_yaml: str | Path | None) -> Dict[int, str]:
    if data_yaml is None:
        return {}
    data = read_yaml(data_yaml)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def resolve_class_ids(names: Dict[int, str], classes: Sequence[str | int] | None) -> List[int]:
    if not classes:
        return []
    out: List[int] = []
    inv = {str(v).lower(): int(k) for k, v in names.items()}
    for c in classes:
        if isinstance(c, int):
            out.append(c)
        else:
            cs = str(c)
            if cs.isdigit():
                out.append(int(cs))
            elif cs.lower() in inv:
                out.append(inv[cs.lower()])
            else:
                raise ValueError(f"Unknown class {c!r}; available: {names}")
    return sorted(set(out))
