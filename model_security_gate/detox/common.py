from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model_security_gate.utils.io import list_images, read_image_bgr, read_yolo_labels
from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback
from model_security_gate.utils.geometry import XYXY


def get_ultralytics_yolo(model_or_path: Any):
    """Return an Ultralytics YOLO wrapper from a path, adapter, or YOLO object."""
    if isinstance(model_or_path, (str, Path)):
        patch_torchvision_nms_fallback()
        from ultralytics import YOLO

        return YOLO(str(model_or_path))
    if hasattr(model_or_path, "model") and hasattr(model_or_path.model, "predict"):
        # UltralyticsYOLOAdapter
        return model_or_path.model
    if hasattr(model_or_path, "predict") and hasattr(model_or_path, "save"):
        return model_or_path
    raise TypeError("Expected a model path, UltralyticsYOLOAdapter, or ultralytics.YOLO object")


def get_torch_model(model_or_path: Any) -> torch.nn.Module:
    yolo = get_ultralytics_yolo(model_or_path)
    if not hasattr(yolo, "model"):
        raise TypeError("Ultralytics YOLO wrapper does not expose .model")
    return yolo.model


def save_yolo(yolo: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(output_path))
    return output_path


def find_ultralytics_weight(project: str | Path, name: str, prefer: str = "best") -> Path:
    weights_dir = Path(project) / name / "weights"
    preferred = weights_dir / f"{prefer}.pt"
    if preferred.exists():
        return preferred
    for cand in [weights_dir / "best.pt", weights_dir / "last.pt"]:
        if cand.exists():
            return cand
    found = sorted(weights_dir.glob("*.pt")) if weights_dir.exists() else []
    if not found:
        raise FileNotFoundError(f"No .pt weights found under {weights_dir}")
    return found[0]


def infer_device(device: str | int | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if device >= 0 else "cpu")
    ds = str(device)
    if ds.lower() in {"cpu", "mps"}:
        return torch.device(ds.lower())
    if ds.isdigit():
        return torch.device(f"cuda:{ds}")
    return torch.device(ds)


def resize_bgr_to_tensor(image_bgr: np.ndarray, imgsz: int = 640) -> torch.Tensor:
    """Simple square resize preprocessing for feature-level detox loops.

    Ultralytics training uses letterbox/augmentations internally. For feature
    alignment and adversarial smoothing, a deterministic square resize is enough
    and keeps the loop dependency-light. Values are normalized to [0, 1].
    """
    img = cv2.resize(image_bgr, (int(imgsz), int(imgsz)), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class ImageTensorDataset(Dataset):
    def __init__(self, image_paths: Sequence[str | Path], imgsz: int = 640) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.imgsz = int(imgsz)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        img = read_image_bgr(path)
        h, w = img.shape[:2]
        return {
            "image": resize_bgr_to_tensor(img, self.imgsz),
            "path": str(path),
            "orig_shape": (h, w),
        }


class LabelledImageTensorDataset(Dataset):
    def __init__(self, image_paths: Sequence[str | Path], labels_dir: str | Path, imgsz: int = 640) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.labels_dir = Path(labels_dir)
        self.imgsz = int(imgsz)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        img = read_image_bgr(path)
        h, w = img.shape[:2]
        labels = read_yolo_labels(path, img.shape, labels_dir=self.labels_dir)
        return {
            "image": resize_bgr_to_tensor(img, self.imgsz),
            "path": str(path),
            "orig_shape": (h, w),
            "labels": labels,
        }


def collate_images(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "orig_shape": [b["orig_shape"] for b in batch],
        "labels": [b.get("labels", []) for b in batch],
    }


def list_conv_modules(torch_model: torch.nn.Module, name_contains: Sequence[str] | None = None) -> List[Tuple[str, torch.nn.Conv2d]]:
    contains = [str(x) for x in (name_contains or []) if str(x)]
    out: List[Tuple[str, torch.nn.Conv2d]] = []
    for name, mod in torch_model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            if contains and not any(c in name for c in contains):
                continue
            out.append((name, mod))
    return out


def select_evenly(items: Sequence[Any], max_items: int) -> List[Any]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    idx = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    return [items[int(i)] for i in idx]


@dataclass
class HookSpec:
    name: str
    module: torch.nn.Module


class FeatureHookBank:
    """Collect forward activations from selected modules for one model."""

    def __init__(self, specs: Sequence[HookSpec], detach: bool = False) -> None:
        self.specs = list(specs)
        self.detach = bool(detach)
        self.features: List[torch.Tensor] = []
        self.handles: List[Any] = []

    def __enter__(self):
        self.clear()
        for idx, spec in enumerate(self.specs):
            self.handles.append(spec.module.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.features = []

    def _make_hook(self, _idx: int):
        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(x) or x.ndim != 4:
                return
            self.features.append(x.detach() if self.detach else x)

        return hook


def pair_feature_hooks(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    max_layers: int = 6,
    name_contains: Sequence[str] | None = None,
) -> Tuple[List[HookSpec], List[HookSpec]]:
    """Choose roughly corresponding Conv2d layers from student and teacher.

    If architectures differ, layers are matched by relative depth, not by name.
    Attention-map distillation only needs spatial maps, so channel counts may differ.
    """
    s_mods = list_conv_modules(student, name_contains)
    t_mods = list_conv_modules(teacher, name_contains)
    s_sel = select_evenly(s_mods, max_layers)
    t_sel = select_evenly(t_mods, max_layers)
    n = min(len(s_sel), len(t_sel))
    return [HookSpec(nm, mod) for nm, mod in s_sel[-n:]], [HookSpec(nm, mod) for nm, mod in t_sel[-n:]]


def attention_map(feat: torch.Tensor, p: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    att = feat.abs().pow(float(p)).mean(dim=1, keepdim=True)
    norm = torch.sqrt(torch.sum(att.pow(2), dim=(2, 3), keepdim=True) + eps)
    return att / norm


def attention_alignment_loss(student_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor], p: float = 2.0) -> torch.Tensor:
    if not student_feats or not teacher_feats:
        raise RuntimeError("No features captured. Check hook layer selection and model forward path.")
    losses: List[torch.Tensor] = []
    for sf, tf in zip(student_feats, teacher_feats):
        sa = attention_map(sf, p=p)
        ta = attention_map(tf.detach(), p=p)
        if sa.shape[-2:] != ta.shape[-2:]:
            sa = torch.nn.functional.interpolate(sa, size=ta.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(torch.nn.functional.mse_loss(sa, ta))
    return torch.stack(losses).mean()


def forward_for_features(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Forward helper; works for Ultralytics DetectionModel in eval/train modes."""
    return model(x)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)


def roi_pool_feature_from_yolo_label(
    feat: torch.Tensor,
    label: Dict[str, Any],
    orig_shape: Tuple[int, int],
) -> Optional[torch.Tensor]:
    """Pool one feature vector from a bbox label on a resized feature map.

    feat shape: C x Hf x Wf for one image. label['xyxy'] is in original pixels.
    """
    c, hf, wf = feat.shape
    oh, ow = orig_shape
    if oh <= 0 or ow <= 0:
        return None
    x1, y1, x2, y2 = label["xyxy"]
    fx1 = int(np.floor(max(0.0, x1 / ow) * wf))
    fx2 = int(np.ceil(min(float(ow), x2) / ow * wf))
    fy1 = int(np.floor(max(0.0, y1 / oh) * hf))
    fy2 = int(np.ceil(min(float(oh), y2) / oh * hf))
    fx1, fx2 = max(0, min(wf - 1, fx1)), max(1, min(wf, fx2))
    fy1, fy2 = max(0, min(hf - 1, fy1)), max(1, min(hf, fy2))
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    return feat[:, fy1:fy2, fx1:fx2].mean(dim=(1, 2))
