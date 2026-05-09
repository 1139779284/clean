from __future__ import annotations

"""Compatibility shims for CPU/lightweight TorchVision installs.

Some CI/handoff environments have PyTorch CPU wheels and a TorchVision package
whose compiled C++ ops are missing. Importing TorchVision can then fail while it
registers fake kernels for ``torchvision::nms``. Ultralytics needs NMS during
prediction, so this module defines a minimal operator schema when necessary and
patches ``torchvision.ops.nms`` with a Python implementation if native NMS is
not available.
"""

import torch

_PATCHED = False
_SCHEMA_DEFINED = False


def _ensure_torchvision_nms_schema() -> None:
    global _SCHEMA_DEFINED
    if _SCHEMA_DEFINED:
        return
    try:
        # If this succeeds, the operator already exists.
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")  # type: ignore[attr-defined]
        _SCHEMA_DEFINED = True
        return
    except Exception:
        pass
    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        # Keep the library object alive by storing it on torch.
        setattr(torch, "_msg_torchvision_nms_def_lib", lib)
    except Exception:
        pass
    _SCHEMA_DEFINED = True


def _python_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        union = (areas[i] + areas[rest] - inter).clamp_min(1e-6)
        iou = inter / union
        order = rest[iou <= float(iou_threshold)]
    return torch.stack(keep).to(dtype=torch.long, device=boxes.device) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def patch_torchvision_nms_fallback() -> bool:
    """Patch ``torchvision.ops.nms`` with a Python fallback if native ops fail.

    Returns True when a fallback was installed. Returns False when native NMS is
    available or TorchVision is not importable even after schema repair.
    """
    global _PATCHED
    if _PATCHED:
        return True
    _ensure_torchvision_nms_schema()
    try:
        import torchvision.ops  # type: ignore
    except Exception:
        return False
    try:
        probe_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        probe_scores = torch.tensor([1.0])
        torchvision.ops.nms(probe_boxes, probe_scores, 0.5)  # type: ignore[attr-defined]
        return False
    except Exception:
        torchvision.ops.nms = _python_nms  # type: ignore[attr-defined]
        _PATCHED = True
        return True
