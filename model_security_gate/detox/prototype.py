from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_security_gate.detox.feature_hooks import ActivationCatcher, select_conv_layers
from model_security_gate.detox.yolo_dataset import move_batch_to_device


@dataclass
class PrototypeBank:
    layer_name: str
    dim: int
    prototypes: Dict[int, torch.Tensor] = field(default_factory=dict)
    counts: Dict[int, int] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "PrototypeBank":
        self.prototypes = {k: v.to(device) for k, v in self.prototypes.items()}
        return self


def _pool_box_feature(feat: torch.Tensor, image_index: int, xywhn: torch.Tensor) -> Optional[torch.Tensor]:
    if feat.ndim != 4:
        return None
    _b, c, h, w = feat.shape
    xc, yc, bw, bh = xywhn.float().tolist()
    x1 = max(0, int((xc - bw / 2.0) * w))
    y1 = max(0, int((yc - bh / 2.0) * h))
    x2 = min(w - 1, int((xc + bw / 2.0) * w))
    y2 = min(h - 1, int((yc + bh / 2.0) * h))
    if x2 < x1 or y2 < y1:
        return None
    region = feat[int(image_index), :, y1:y2 + 1, x1:x2 + 1]
    if region.numel() == 0:
        return None
    vec = region.mean(dim=(1, 2))
    return F.normalize(vec.float(), dim=0)


def build_prototype_bank(
    model: torch.nn.Module,
    loader,
    layer_name: Optional[str] = None,
    max_batches: int = 50,
    device: str | torch.device = "cpu",
) -> PrototypeBank:
    device = torch.device(device)
    layer_names = [layer_name] if layer_name else select_conv_layers(model, max_layers=1, prefer_late=True)
    if not layer_names:
        raise ValueError("No layer available for prototype extraction")
    layer = layer_names[0]
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Build prototypes")):
            if bi >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            with ActivationCatcher(model, [layer]) as ac:
                _ = model(batch["img"])
            feat = ac.features.get(layer)
            if feat is None:
                continue
            cls = batch["cls"].view(-1).long()
            bboxes = batch["bboxes"]
            bidx = batch["batch_idx"].long()
            for j in range(len(cls)):
                pooled = _pool_box_feature(feat, int(bidx[j].item()), bboxes[j])
                if pooled is None:
                    continue
                cid = int(cls[j].item())
                if cid not in sums:
                    sums[cid] = pooled.detach().clone()
                    counts[cid] = 1
                else:
                    sums[cid] += pooled.detach()
                    counts[cid] += 1
    model.train(was_training)
    if not sums:
        # Make a harmless zero-dimensional bank rather than crashing a pipeline.
        return PrototypeBank(layer_name=layer, dim=0, prototypes={}, counts={})
    protos = {cid: F.normalize(vec / max(1, counts[cid]), dim=0) for cid, vec in sums.items()}
    dim = int(next(iter(protos.values())).numel())
    return PrototypeBank(layer_name=layer, dim=dim, prototypes=protos, counts=counts)


def prototype_alignment_loss(
    features: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    bank: PrototypeBank,
    margin: float = 0.2,
) -> torch.Tensor:
    feat = features.get(bank.layer_name)
    if feat is None or not bank.prototypes:
        return torch.tensor(0.0, device=batch["img"].device)
    losses: List[torch.Tensor] = []
    cls = batch["cls"].view(-1).long()
    bboxes = batch["bboxes"]
    bidx = batch["batch_idx"].long()
    for j in range(len(cls)):
        cid = int(cls[j].item())
        proto = bank.prototypes.get(cid)
        if proto is None:
            continue
        pooled = _pool_box_feature(feat, int(bidx[j].item()), bboxes[j])
        if pooled is None:
            continue
        proto = proto.to(pooled.device)
        pos = 1.0 - F.cosine_similarity(pooled[None], proto[None], dim=1).mean()
        # Push away from nearest unrelated prototype with a small margin.
        neg_terms = []
        for ncid, nproto in bank.prototypes.items():
            if ncid == cid:
                continue
            nproto = nproto.to(pooled.device)
            sim_neg = F.cosine_similarity(pooled[None], nproto[None], dim=1).mean()
            sim_pos = F.cosine_similarity(pooled[None], proto[None], dim=1).mean()
            neg_terms.append(F.relu(float(margin) + sim_neg - sim_pos))
        if neg_terms:
            losses.append(pos + torch.stack(neg_terms).mean())
        else:
            losses.append(pos)
    if not losses:
        return torch.tensor(0.0, device=batch["img"].device)
    return torch.stack(losses).mean()
