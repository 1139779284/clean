from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def get_torch_model(obj: Any) -> torch.nn.Module:
    """Extract the underlying torch module from common wrappers."""
    if isinstance(obj, torch.nn.Module):
        return obj
    if hasattr(obj, "model") and isinstance(getattr(obj, "model"), torch.nn.Module):
        return obj.model
    if hasattr(obj, "model") and hasattr(obj.model, "model") and isinstance(obj.model.model, torch.nn.Module):
        return obj.model.model
    raise TypeError("Could not extract a torch.nn.Module from object")


def module_dict(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def select_conv_layers(
    model: torch.nn.Module,
    contains: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    max_layers: int = 8,
    prefer_late: bool = True,
) -> List[str]:
    contains = list(contains or [])
    exclude = list(exclude or [])
    names: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Conv2d):
            continue
        if contains and not any(s in name for s in contains):
            continue
        if exclude and any(s in name for s in exclude):
            continue
        names.append(name)
    if max_layers and len(names) > max_layers:
        if prefer_late:
            names = names[-max_layers:]
        else:
            idx = torch.linspace(0, len(names) - 1, steps=max_layers).round().long().tolist()
            names = [names[i] for i in idx]
    return names


class ActivationCatcher:
    """Forward-hook activations for a selected list of module names.

    If retain_grad=True, the captured tensors retain gradients after backward;
    this is useful for ANP-style channel scoring. The hook does not detach
    tensors, so training losses can be computed from captured features.
    """

    def __init__(self, model: torch.nn.Module, layer_names: Sequence[str], retain_grad: bool = False) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.retain_grad = bool(retain_grad)
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            feat = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(feat):
                if self.retain_grad and feat.requires_grad:
                    feat.retain_grad()
                self.features[name] = feat
        return hook

    def __enter__(self) -> "ActivationCatcher":
        mods = module_dict(self.model)
        for name in self.layer_names:
            mod = mods.get(name)
            if mod is not None:
                self.handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def attention_map(feat: torch.Tensor, power: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError(f"Expected BCHW feature, got {tuple(feat.shape)}")
    attn = feat.abs().pow(power).mean(dim=1, keepdim=True)
    flat = attn.flatten(1)
    denom = flat.sum(dim=1, keepdim=True).clamp_min(eps)
    return (flat / denom).view_as(attn)


def normalized_attention(feat: torch.Tensor, power: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    a = feat.abs().pow(power).mean(dim=1, keepdim=True)
    mean = a.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
    std = a.flatten(1).std(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
    return (a - mean) / std


def recursive_tensor_pairs(a: Any, b: Any) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape == b.shape and a.numel() > 0:
            yield a, b
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for x, y in zip(a, b):
            yield from recursive_tensor_pairs(x, y)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a.keys()).intersection(b.keys()):
            yield from recursive_tensor_pairs(a[k], b[k])


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            t = _first_tensor(x)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for x in obj.values():
            t = _first_tensor(x)
            if t is not None:
                return t
    return None


def output_distillation_loss(student_out: Any, teacher_out: Any, mode: str = "mse") -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for s, t in recursive_tensor_pairs(student_out, teacher_out):
        t = t.detach()
        if mode == "smooth_l1":
            losses.append(F.smooth_l1_loss(s.float(), t.float()))
        else:
            # YOLO raw outputs are not probabilities; MSE is stable across versions.
            losses.append(F.mse_loss(s.float(), t.float()))
    if not losses:
        ref = _first_tensor(student_out)
        if ref is None:
            ref = _first_tensor(teacher_out)
        if ref is None:
            return torch.tensor(0.0)
        return ref.float().sum() * 0.0
    return torch.stack(losses).mean()


def feature_distillation_loss(student_feats: Dict[str, torch.Tensor], teacher_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for name, s in student_feats.items():
        t = teacher_feats.get(name)
        if t is None or s.ndim != 4 or t.ndim != 4:
            continue
        if s.shape[-2:] != t.shape[-2:]:
            t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        if s.shape[1] != t.shape[1]:
            c = min(s.shape[1], t.shape[1])
            s2, t2 = s[:, :c], t[:, :c]
        else:
            s2, t2 = s, t
        losses.append(F.mse_loss(s2.float(), t2.detach().float()))
    if not losses:
        return torch.tensor(0.0, device=next(iter(student_feats.values())).device) if student_feats else torch.tensor(0.0)
    return torch.stack(losses).mean()


def nad_attention_loss(student_feats: Dict[str, torch.Tensor], teacher_feats: Dict[str, torch.Tensor], power: float = 2.0) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for name, s in student_feats.items():
        t = teacher_feats.get(name)
        if t is None or s.ndim != 4 or t.ndim != 4:
            continue
        sa = normalized_attention(s, power=power)
        ta = normalized_attention(t.detach(), power=power)
        if sa.shape[-2:] != ta.shape[-2:]:
            ta = F.interpolate(ta, size=sa.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(sa, ta))
    if not losses:
        return torch.tensor(0.0, device=next(iter(student_feats.values())).device) if student_feats else torch.tensor(0.0)
    return torch.stack(losses).mean()


def bbox_union_mask_from_batch(
    batch: Dict[str, Any],
    image_index: int,
    out_hw: Tuple[int, int],
    class_ids: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    h, w = out_hw
    device = device or batch["img"].device
    mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    if "batch_idx" not in batch or "bboxes" not in batch or "cls" not in batch:
        return mask
    bidx = batch["batch_idx"].long()
    sel = bidx == int(image_index)
    if class_ids is not None:
        wanted = torch.tensor([int(c) for c in class_ids], device=device)
        cls = batch["cls"].view(-1).long()
        sel = sel & (cls[:, None] == wanted[None, :]).any(dim=1)
    boxes = batch["bboxes"][sel]
    for box in boxes:
        xc, yc, bw, bh = box.float().tolist()
        x1 = max(0, int((xc - bw / 2.0) * w))
        y1 = max(0, int((yc - bh / 2.0) * h))
        x2 = min(w - 1, int((xc + bw / 2.0) * w))
        y2 = min(h - 1, int((yc + bh / 2.0) * h))
        if x2 >= x1 and y2 >= y1:
            mask[:, y1:y2 + 1, x1:x2 + 1] = 1.0
    return mask
