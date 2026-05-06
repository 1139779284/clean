from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model_security_gate.utils.io import IMAGE_EXTS, read_yaml


@dataclass
class YoloDatasetInfo:
    data_yaml: Path
    root: Path
    names: Dict[int, str]
    train_images: List[Path]
    val_images: List[Path]


def _resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _expand_image_source(root: Path, source: Any) -> List[Path]:
    if source is None:
        return []
    if isinstance(source, (list, tuple)):
        out: List[Path] = []
        for x in source:
            out.extend(_expand_image_source(root, x))
        return sorted(dict.fromkeys(out))
    p = _resolve_path(root, str(source))
    if p.is_file() and p.suffix.lower() == ".txt":
        out: List[Path] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ip = Path(line)
            if not ip.is_absolute():
                ip = (p.parent / ip).resolve()
            if ip.exists() and ip.suffix.lower() in IMAGE_EXTS:
                out.append(ip)
        return sorted(dict.fromkeys(out))
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS])
    return []


def parse_yolo_data_yaml(data_yaml: str | Path) -> YoloDatasetInfo:
    data_yaml = Path(data_yaml).resolve()
    cfg = read_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names_raw = cfg.get("names", {})
    if isinstance(names_raw, list):
        names = {i: str(v) for i, v in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        names = {}
    return YoloDatasetInfo(
        data_yaml=data_yaml,
        root=root,
        names=names,
        train_images=_expand_image_source(root, cfg.get("train")),
        val_images=_expand_image_source(root, cfg.get("val")),
    )


def image_to_label_path(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        # Use the last occurrence, so nested folders work.
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def _letterbox_or_resize(image_bgr: np.ndarray, imgsz: int, letterbox: bool = False) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Resize image for training.

    Default is direct resize, because YOLO normalized xywh labels remain valid
    under direct H/W scaling. Letterbox is available for users who prefer less
    distortion, but direct resize is simpler and robust for detox fine-tuning.
    """
    h, w = image_bgr.shape[:2]
    if not letterbox:
        resized = cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        return resized, (imgsz / max(w, 1), imgsz / max(h, 1)), (0.0, 0.0)
    scale = min(imgsz / max(h, 1), imgsz / max(w, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dx = (imgsz - nw) // 2
    dy = (imgsz - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, (scale, scale), (float(dx), float(dy))


def _read_normalized_labels(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    cls: List[float] = []
    boxes: List[List[float]] = []
    if not label_path.exists():
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        c = float(parts[0])
        vals = [float(x) for x in parts[1:5]]
        # Clamp malformed labels rather than crashing the detox loop.
        vals = [float(np.clip(v, 0.0, 1.0)) for v in vals]
        if vals[2] <= 0.0 or vals[3] <= 0.0:
            continue
        cls.append(c)
        boxes.append(vals)
    return np.asarray(cls, dtype=np.float32).reshape(-1, 1), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class YoloDetoxDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str | Path],
        imgsz: int = 640,
        max_images: Optional[int] = None,
        letterbox: bool = False,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        if max_images is not None and max_images > 0:
            self.image_paths = self.image_paths[:max_images]
        self.imgsz = int(imgsz)
        self.letterbox = bool(letterbox)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        orig_shape = img.shape[:2]
        img, _scale, _pad = _letterbox_or_resize(img, self.imgsz, self.letterbox)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().contiguous() / 255.0
        cls_np, boxes_np = _read_normalized_labels(image_to_label_path(path))
        return {
            "img": img_t,
            "cls": torch.from_numpy(cls_np),
            "bboxes": torch.from_numpy(boxes_np),
            "im_file": str(path),
            "ori_shape": orig_shape,
            "resized_shape": (self.imgsz, self.imgsz),
        }


def yolo_collate_fn(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    imgs = torch.stack([s["img"] for s in samples], dim=0)
    cls_parts: List[torch.Tensor] = []
    bbox_parts: List[torch.Tensor] = []
    batch_idx_parts: List[torch.Tensor] = []
    for i, s in enumerate(samples):
        c = s["cls"].float().reshape(-1, 1)
        b = s["bboxes"].float().reshape(-1, 4)
        if len(c):
            cls_parts.append(c)
            bbox_parts.append(b)
            batch_idx_parts.append(torch.full((len(c),), i, dtype=torch.float32))
    if cls_parts:
        cls = torch.cat(cls_parts, dim=0)
        bboxes = torch.cat(bbox_parts, dim=0)
        batch_idx = torch.cat(batch_idx_parts, dim=0)
    else:
        cls = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.float32)
    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": [s["im_file"] for s in samples],
        "ori_shape": [s["ori_shape"] for s in samples],
        "resized_shape": [s["resized_shape"] for s in samples],
    }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_yolo_dataloader(
    data_yaml: str | Path,
    split: str = "train",
    imgsz: int = 640,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_images: Optional[int] = None,
    letterbox: bool = False,
) -> Tuple[DataLoader, YoloDatasetInfo]:
    info = parse_yolo_data_yaml(data_yaml)
    paths = info.train_images if split == "train" else info.val_images
    if not paths:
        raise FileNotFoundError(f"No {split} images found in {data_yaml}")
    ds = YoloDetoxDataset(paths, imgsz=imgsz, max_images=max_images, letterbox=letterbox)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available() and int(num_workers) > 0,
        collate_fn=yolo_collate_fn,
        drop_last=False,
    )
    return loader, info
