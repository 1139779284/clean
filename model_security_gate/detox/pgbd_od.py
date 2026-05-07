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


def make_pgbd_attack_view(
    img: torch.Tensor,
    mode: str = "mixed",
    green_strength: float = 0.35,
    blend_alpha: float = 0.10,
    warp_amplitude: float = 0.025,
) -> torch.Tensor:
    mode = str(mode or "mixed").lower()
    out = img
    if mode in {"green", "semantic", "semantic_green"}:
        return semantic_green_view(out, strength=green_strength)
    if mode in {"blend", "sinusoidal"}:
        return sinusoidal_blend_view(out, alpha=blend_alpha)
    if mode in {"warp", "wanet", "smooth_warp"}:
        return smooth_warp_view(out, amplitude=warp_amplitude)
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
