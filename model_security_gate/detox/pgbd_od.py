from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from model_security_gate.detox.prototype import PrototypeBank, _pool_box_feature  # type: ignore


def _target_image_mask(batch: Dict[str, Any], image_index: int, target_class_ids: Sequence[int], device: torch.device) -> bool:
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    if cls.numel() == 0 or not target_class_ids:
        return False
    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long)
    return bool(((bidx == image_index) & (cls[:, None] == target_ids[None, :]).any(dim=1)).any())


def _attention_global_pool(feat: torch.Tensor) -> torch.Tensor:
    attn = feat.abs().mean(dim=1, keepdim=True)
    attn = attn / attn.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    pooled = (feat.float() * attn).sum(dim=(2, 3))
    return F.normalize(pooled, dim=1)


def semantic_green_view(img: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    """Differentiable semantic-green style perturbation for RGB/BGR-agnostic tensors.

    It boosts the middle channel and weakly suppresses the other channels. The
    exact color space is less important than creating a semantic/color shortcut
    pressure during paired consistency training.
    """
    out = img.clone()
    if out.ndim != 4 or out.shape[1] < 3:
        return out
    green = torch.zeros_like(out)
    green[:, 1:2] = 0.85
    green[:, 0:1] = 0.15
    green[:, 2:3] = 0.15
    # Apply more strongly to non-dark pixels to avoid destroying empty padding.
    mask = (img.mean(dim=1, keepdim=True) > 0.05).float()
    out = img * (1.0 - float(strength) * mask) + green * (float(strength) * mask)
    return out.clamp(0.0, 1.0)


def sinusoidal_blend_view(img: torch.Tensor, alpha: float = 0.10, freq: float = 6.0) -> torch.Tensor:
    if img.ndim != 4:
        return img
    b, c, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(0, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    wave = (torch.sin(2.0 * math.pi * float(freq) * xx) + torch.cos(2.0 * math.pi * float(freq) * yy)) * 0.25 + 0.5
    pattern = wave.view(1, 1, h, w).repeat(b, c, 1, 1)
    return (img * (1.0 - float(alpha)) + pattern * float(alpha)).clamp(0.0, 1.0)


def smooth_warp_view(img: torch.Tensor, amplitude: float = 0.025) -> torch.Tensor:
    """Small deterministic smooth warp using grid_sample.

    This is intentionally mild; it supplies paired WaNet-like pressure without
    turning detox training into a new distribution.
    """
    if img.ndim != 4:
        return img
    b, _c, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    flow_x = torch.sin(math.pi * 3.0 * yy) * float(amplitude)
    flow_y = torch.cos(math.pi * 3.0 * xx) * float(amplitude)
    grid = torch.stack([xx + flow_x, yy + flow_y], dim=-1).view(1, h, w, 2).repeat(b, 1, 1, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True).clamp(0.0, 1.0)


def badnet_patch_view(
    img: torch.Tensor,
    *,
    patch_frac: float = 0.06,
    placement: str = "object_attached",
    box_xyxy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable BadNet-style patch overlay for PGBD paired training.

    Unlike ``semantic_green_view`` / ``sinusoidal_blend_view`` / ``smooth_warp_view``
    (which are global), this view stamps a small checkerboard patch on top of
    the image so the attacked view matches the ``attack_zoo`` BadNet evaluation
    trigger.  Two placements are supported:

    * ``"object_attached"`` uses the supplied ``box_xyxy`` (B, 4) pixel bboxes
      to place the patch near the top-right of each target box.  When no box is
      supplied for an image, the patch falls back to ``"bottom_right"`` on that
      image so the op never silently becomes a no-op.
    * ``"bottom_right"`` always stamps in the bottom-right corner regardless of
      boxes.  This mirrors the ``attack_zoo`` BadNet OGA default.

    The patch is written as a deterministic 2x2 checker pattern in ``[0, 1]``.
    A small blend (alpha=0.96) keeps some gradient from the underlying image so
    the feature extractor is pushed rather than clamped.
    """
    if img.ndim != 4 or img.shape[1] < 1:
        return img
    out = img.clone()
    b, _c, h, w = out.shape
    side = max(4, int(round(min(h, w) * float(patch_frac))))
    # Deterministic 2x2 checker tile, repeated to fill the patch.
    tile = max(2, side // 4)
    yy, xx = torch.meshgrid(
        torch.arange(side, device=out.device),
        torch.arange(side, device=out.device),
        indexing="ij",
    )
    mask = (((xx // tile) + (yy // tile)) % 2) == 0
    patch_val = torch.zeros((out.shape[1], side, side), device=out.device, dtype=out.dtype)
    patch_val[:, mask] = 1.0  # white
    # Non-mask region stays 0 (black), giving a BadNet-like checker.
    blend = 0.96

    for i in range(b):
        if str(placement).lower() == "object_attached" and box_xyxy is not None and box_xyxy.numel() >= (i + 1) * 4:
            x1, y1, x2, _y2 = [float(v) for v in box_xyxy.view(-1, 4)[i].tolist()]
            # attack_zoo places the patch near the top-right of the helmet.
            px = int(min(max(0.0, x2 - side * 0.65), max(0.0, w - side)))
            py = int(min(max(0.0, y1 - side * 0.35), max(0.0, h - side)))
        else:
            px, py = max(0, w - side - 2), max(0, h - side - 2)
        py2 = min(h, py + side)
        px2 = min(w, px + side)
        ph = py2 - py
        pw = px2 - px
        if ph <= 0 or pw <= 0:
            continue
        out[i, :, py:py2, px:px2] = (
            blend * patch_val[:, :ph, :pw] + (1.0 - blend) * out[i, :, py:py2, px:px2]
        )
    return out.clamp(0.0, 1.0)


_PHASE_TO_VIEW_MODE = {
    "oga_hardening": "badnet",  # BadNet OGA = corner patch
    "oda_hardening": "badnet_object_attached",  # BadNet ODA = patch on helmet
    "wanet_hardening": "warp",
    "semantic_hardening": "green",
    "clean_anchor": "mixed",
    "clean_recovery": "mixed",
}


def infer_pgbd_mode_from_phase(phase_name: str | None, default: str = "mixed") -> str:
    """Pick a PGBD attack view mode that matches the detox phase.

    Earlier versions used a single ``mixed`` view (green + sinusoidal + warp)
    for every phase.  That mismatched the real BadNet evaluation trigger and
    the warp amplitude did not match the attack_zoo ``wanet_oga`` strength, so
    the paired displacement loss pushed features against an irrelevant
    perturbation.  ``infer_pgbd_mode_from_phase`` selects a view that mirrors
    the corresponding ``attack_zoo`` family.
    """
    if not phase_name:
        return default
    low = str(phase_name).lower()
    for key, mode in _PHASE_TO_VIEW_MODE.items():
        if key in low:
            return mode
    # Fall back to the explicit tokens used by phase planner variations.
    if "oda" in low:
        return "badnet_object_attached"
    if "oga" in low:
        return "badnet"
    if "wanet" in low or "warp" in low:
        return "warp"
    if "semantic" in low:
        return "green"
    return default


def make_pgbd_attack_view(
    img: torch.Tensor,
    mode: str = "mixed",
    green_strength: float = 0.35,
    blend_alpha: float = 0.10,
    warp_amplitude: float = 0.025,
    badnet_patch_frac: float = 0.06,
    badnet_box_xyxy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a paired attack view for PGBD displacement training.

    Supported modes:

    * ``"green" / "semantic" / "semantic_green"`` - differentiable green-vest
      overlay (mirrors ``semantic_green_cleanlabel``).
    * ``"blend" / "sinusoidal"`` - sinusoidal blend (mirrors ``blend_oga``).
    * ``"warp" / "wanet" / "smooth_warp"`` - small smooth warp (mirrors
      ``wanet_oga``).
    * ``"badnet"`` - checker patch stamped in the bottom-right corner
      (mirrors ``badnet_oga_corner``).
    * ``"badnet_object_attached"`` - checker patch stamped on the supplied
      bounding box (mirrors ``badnet_oda_object``).  When no box is supplied
      this falls back to bottom-right placement so the op is never a no-op.
    * ``"mixed"`` - mild composition of green + blend + warp, preserved for
      backward compatibility.
    """
    mode = str(mode or "mixed").lower()
    out = img
    if mode in {"green", "semantic", "semantic_green"}:
        return semantic_green_view(out, strength=green_strength)
    if mode in {"blend", "sinusoidal"}:
        return sinusoidal_blend_view(out, alpha=blend_alpha)
    if mode in {"warp", "wanet", "smooth_warp"}:
        return smooth_warp_view(out, amplitude=warp_amplitude)
    if mode == "badnet":
        return badnet_patch_view(out, patch_frac=badnet_patch_frac, placement="bottom_right")
    if mode in {"badnet_object_attached", "badnet_oda", "badnet_object"}:
        return badnet_patch_view(
            out,
            patch_frac=badnet_patch_frac,
            placement="object_attached",
            box_xyxy=badnet_box_xyxy,
        )
    # Mixed view: mild composition, useful as a generic unknown-trigger pressure.
    out = semantic_green_view(out, strength=green_strength)
    out = sinusoidal_blend_view(out, alpha=blend_alpha)
    out = smooth_warp_view(out, amplitude=warp_amplitude)
    return out.clamp(0.0, 1.0)


def pgbd_paired_displacement_loss(
    clean_features: Dict[str, torch.Tensor],
    attacked_features: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    bank: PrototypeBank | None,
    target_class_ids: Sequence[int],
    *,
    layer_name: str | None = None,
    target_weight: float = 1.0,
    negative_weight: float = 1.0,
    displacement_weight: float = 0.50,
    negative_margin: float = 0.25,
) -> torch.Tensor:
    """PGBD-OD paired clean/attacked activation displacement loss.

    For target-present boxes: attacked ROI features should stay close to the
    clean/teacher ROI feature and to the class prototype.

    For target-absent images: attacked global evidence should not move toward a
    target prototype. This directly targets semantic/trigger shortcuts where a
    background or color context becomes helmet-like.
    """
    if bank is None or not bank.prototypes:
        ref = batch.get("img")
        return ref.sum() * 0.0 if torch.is_tensor(ref) else torch.tensor(0.0)
    layer = layer_name or bank.layer_name
    clean = clean_features.get(layer)
    attacked = attacked_features.get(layer)
    if clean is None or attacked is None or clean.ndim != 4 or attacked.ndim != 4:
        ref = batch.get("img")
        return ref.sum() * 0.0 if torch.is_tensor(ref) else torch.tensor(0.0)
    device = attacked.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long) if target_class_ids else torch.empty((0,), device=device, dtype=torch.long)
    losses: List[torch.Tensor] = []

    # Positive/target-present ROI consistency and prototype alignment.
    if cls.numel() and target_ids.numel():
        target_label_mask = (cls[:, None] == target_ids[None, :]).any(dim=1)
        for j in torch.where(target_label_mask)[0].tolist():
            cid = int(cls[j].item())
            proto = bank.prototypes.get(cid)
            if proto is None:
                continue
            clean_vec = _pool_box_feature(clean, int(bidx[j].item()), bboxes[j])
            attack_vec = _pool_box_feature(attacked, int(bidx[j].item()), bboxes[j])
            if clean_vec is None or attack_vec is None:
                continue
            clean_vec = clean_vec.detach().to(device)
            attack_vec = attack_vec.to(device)
            proto = F.normalize(proto.to(device).float(), dim=0)
            pair = 1.0 - F.cosine_similarity(attack_vec[None], clean_vec[None], dim=1).mean()
            proto_align = 1.0 - F.cosine_similarity(attack_vec[None], proto[None], dim=1).mean()
            # Displacement should not point away from clean feature and toward a
            # wrong direction; keep it small in target ROI.
            disp = F.normalize((attack_vec - clean_vec).float(), dim=0)
            proto_dir = F.normalize((proto - clean_vec).float(), dim=0)
            disp_toward_proto = F.relu(F.cosine_similarity(disp[None], proto_dir[None], dim=1).mean())
            losses.append(float(target_weight) * (pair + proto_align + float(displacement_weight) * disp_toward_proto))

    # Negative/target-absent suppression of target-prototype drift.
    target_protos = [bank.prototypes.get(int(cid)) for cid in target_class_ids if int(cid) in bank.prototypes]
    target_protos = [F.normalize(p.to(device).float(), dim=0) for p in target_protos if p is not None]
    if target_protos:
        clean_pool = _attention_global_pool(clean)
        attack_pool = _attention_global_pool(attacked)
        for i in range(attacked.shape[0]):
            if _target_image_mask(batch, i, target_class_ids, device):
                continue
            delta = F.normalize((attack_pool[i] - clean_pool[i]).float(), dim=0)
            terms = []
            for proto in target_protos:
                attack_sim = F.cosine_similarity(attack_pool[i : i + 1], proto.view(1, -1), dim=1).mean()
                delta_sim = F.cosine_similarity(delta.view(1, -1), proto.view(1, -1), dim=1).mean()
                terms.append(F.relu(attack_sim - float(negative_margin)) + F.relu(delta_sim))
            if terms:
                losses.append(float(negative_weight) * torch.stack(terms).mean())

    if not losses:
        return attacked.sum() * 0.0
    return torch.stack(losses).mean()
