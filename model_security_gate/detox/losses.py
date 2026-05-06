from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model_security_gate.detox.feature_hooks import attention_map, bbox_union_mask_from_batch


def _zero_like_model_loss(model: torch.nn.Module) -> torch.Tensor:
    p = next(model.parameters(), None)
    if p is None:
        return torch.tensor(0.0)
    return p.sum() * 0.0


def _ensure_ultralytics_loss_hyp(model: torch.nn.Module) -> None:
    """Make exported Ultralytics models usable with DetectionModel.loss().

    Some `.pt` files restore `model.args` as a plain dict containing only a few
    training keys. Ultralytics' loss path expects attribute-style hyperparameters
    such as `hyp.box`, `hyp.cls`, and `hyp.dfl`. Without this guard, custom detox
    training crashes before doing any useful work.
    """
    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "label_smoothing": 0.0,
    }
    args = getattr(model, "args", None)
    if isinstance(args, Mapping):
        data = dict(defaults)
        data.update(dict(args))
        model.args = SimpleNamespace(**data)
    elif args is None or not all(hasattr(args, key) for key in ("box", "cls", "dfl")):
        data = dict(defaults)
        if args is not None and hasattr(args, "__dict__"):
            data.update(vars(args))
        model.args = SimpleNamespace(**data)

    criterion = getattr(model, "criterion", None)
    if criterion is None and hasattr(model, "init_criterion"):
        try:
            criterion = model.init_criterion()
            model.criterion = criterion
        except Exception:
            criterion = getattr(model, "criterion", None)
    hyp = getattr(criterion, "hyp", None)
    if isinstance(hyp, Mapping):
        data = dict(defaults)
        data.update(dict(hyp))
        criterion.hyp = SimpleNamespace(**data)
    if criterion is not None:
        try:
            device = next(model.parameters()).device
            if hasattr(criterion, "proj") and torch.is_tensor(criterion.proj):
                criterion.proj = criterion.proj.to(device)
        except StopIteration:
            pass


def supervised_yolo_loss(model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """Return Ultralytics DetectionModel supervised loss from a batch dict.

    Ultralytics DetectionModel.forward(batch_dict) returns either a scalar loss
    or a tuple whose first element is the scalar loss. This wrapper normalizes
    those variants and keeps a safe fallback for dry runs.
    """
    _ensure_ultralytics_loss_hyp(model)
    out = model(batch)
    if torch.is_tensor(out):
        return out.mean()
    if isinstance(out, (tuple, list)) and out:
        first = out[0]
        if torch.is_tensor(first):
            return first.mean()
        tensors = [x.mean() for x in out if torch.is_tensor(x)]
        if tensors:
            return torch.stack(tensors).sum()
    if isinstance(out, dict):
        tensors = [v.mean() for v in out.values() if torch.is_tensor(v)]
        if tensors:
            return torch.stack(tensors).sum()
    return _zero_like_model_loss(model)


def clone_batch_with_img(batch: Dict[str, Any], img: torch.Tensor) -> Dict[str, Any]:
    out = dict(batch)
    out["img"] = img
    return out


def pgd_adversarial_images(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    eps: float = 4.0 / 255.0,
    alpha: Optional[float] = None,
    steps: int = 2,
    random_start: bool = True,
) -> torch.Tensor:
    """I-BAU-style inner maximization on the supervised detection loss.

    This does not need to know the true trigger. It asks: what small input
    perturbation most destabilizes the current detector on clean/counterfactual
    labels? The outer detox step then trains the model to resist that shift.
    """
    was_training = model.training
    model.eval()
    x0 = batch["img"].detach()
    if alpha is None:
        alpha = eps / max(1, steps) * 1.5
    if random_start:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x0)
    adv = (x0 + delta).clamp(0.0, 1.0).detach()
    for _ in range(max(1, int(steps))):
        adv.requires_grad_(True)
        adv_batch = clone_batch_with_img(batch, adv)
        loss = supervised_yolo_loss(model, adv_batch)
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            break
        adv = adv.detach() + float(alpha) * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0.0, 1.0).detach()
    model.train(was_training)
    return adv.detach()


def attention_localization_loss(
    features: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    outside_weight: float = 1.0,
) -> torch.Tensor:
    """Encourage target-class evidence to live inside target boxes.

    For images that contain target labels, penalize attention mass outside the
    union of target boxes. For images without target labels, no penalty is added
    here; target suppression is handled by normal detection labels and target
    removal counterfactual samples.
    """
    losses: List[torch.Tensor] = []
    if not features:
        return torch.tensor(0.0, device=batch["img"].device)
    for _name, feat in features.items():
        if feat.ndim != 4:
            continue
        attn = attention_map(feat)
        b, _c, h, w = attn.shape
        for i in range(b):
            mask = bbox_union_mask_from_batch(batch, i, (h, w), class_ids=target_class_ids, device=attn.device)
            if mask.sum() <= 0:
                continue
            inside = (attn[i] * mask).sum()
            total = attn[i].sum().clamp_min(1e-6)
            outside = 1.0 - inside / total
            losses.append(outside * float(outside_weight))
    if not losses:
        return torch.tensor(0.0, device=batch["img"].device)
    return torch.stack(losses).mean()


def consistency_loss_between_outputs(out_a: Any, out_b: Any) -> torch.Tensor:
    from model_security_gate.detox.feature_hooks import output_distillation_loss

    return output_distillation_loss(out_a, out_b, mode="smooth_l1")


def raw_prediction(model: torch.nn.Module, img: torch.Tensor) -> Any:
    return model(img)
