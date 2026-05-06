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


def _find_decoded_prediction(obj: Any) -> Optional[torch.Tensor]:
    """Return a decoded YOLO prediction tensor shaped BCH or BHC.

    Ultralytics YOLOv8/YOLO11 detection models commonly return a tuple whose
    first tensor is ``(B, 4 + nc, N)`` during inference-style forwards. Some
    versions expose ``(B, N, 4 + nc)``. This helper avoids depending on one exact
    wrapper version while keeping the ODA loss optional and safe.
    """
    if torch.is_tensor(obj) and obj.ndim == 3:
        if obj.shape[1] >= 5 or obj.shape[2] >= 5:
            return obj
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_decoded_prediction(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _find_decoded_prediction(item)
            if found is not None:
                return found
    return None


def _prediction_channels_first(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D prediction tensor, got {tuple(pred.shape)}")
    # Prefer the compact channel dimension as C. For YOLO detection C is usually
    # 4 + number_of_classes and N is thousands of candidates.
    if pred.shape[1] >= 5 and pred.shape[2] < 5:
        return pred
    if pred.shape[2] >= 5 and pred.shape[1] < 5:
        return pred.transpose(1, 2).contiguous()
    if pred.shape[1] <= pred.shape[2]:
        return pred
    return pred.transpose(1, 2).contiguous()


def _xywh_to_xyxy_pixels(boxes_xywh: torch.Tensor, img_w: float, img_h: float) -> torch.Tensor:
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:].clamp_min(0.0)
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    out = torch.cat([x1y1, x2y2], dim=1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(img_w))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(img_h))
    return out


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp_min(0.0) * (a[:, 3] - a[:, 1]).clamp_min(0.0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp_min(0.0) * (b[:, 3] - b[:, 1]).clamp_min(0.0))[None, :]
    return inter / (area_a + area_b - inter).clamp_min(1e-6)


def target_recall_confidence_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    min_conf: float = 0.45,
    iou_threshold: float = 0.05,
    center_radius: float = 1.50,
    topk: int = 24,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """ODA recall-preserving loss for target-present images.

    ODA/backdoor disappearance failures usually keep the image visually valid
    while suppressing the target class near a real object. Supervised YOLO loss
    alone can be too diffuse in a mixed detox batch, so this loss adds a direct
    constraint: for every ground-truth target box, at least one decoded candidate
    near that box must keep target confidence above ``min_conf``.

    The matching mask is intentionally broad (IoU OR center-inside an expanded
    GT region), then the loss only backpropagates through class confidence. This
    makes it stable across YOLO versions and avoids fighting the box regressor.
    """
    pred = _find_decoded_prediction(prediction)
    if pred is None:
        ref = batch.get("img")
        return ref.sum() * 0.0 if torch.is_tensor(ref) else torch.tensor(0.0)
    pred = _prediction_channels_first(pred).float()
    if pred.shape[1] < 5 or not target_class_ids:
        return pred.sum() * 0.0

    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    if cls.numel() == 0 or bboxes.numel() == 0:
        return pred.sum() * 0.0

    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long)
    target_sel = (cls[:, None] == target_ids[None, :]).any(dim=1)
    if not bool(target_sel.any()):
        return pred.sum() * 0.0

    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    losses: List[torch.Tensor] = []
    min_conf_t = torch.tensor(float(min_conf), device=device, dtype=pred.dtype)
    topk = max(1, int(topk))
    nc = pred.shape[1] - 4

    for label_index in torch.where(target_sel)[0].tolist():
        image_index = int(bidx[label_index].item())
        cid = int(cls[label_index].item())
        class_channel = 4 + cid
        if image_index < 0 or image_index >= pred.shape[0] or cid < 0 or cid >= nc:
            continue

        gt_xywhn = bboxes[label_index]
        gt_xc = gt_xywhn[0] * img_w
        gt_yc = gt_xywhn[1] * img_h
        gt_w = gt_xywhn[2].clamp_min(1e-6) * img_w
        gt_h = gt_xywhn[3].clamp_min(1e-6) * img_h
        gt_xyxy = torch.stack(
            [
                gt_xc - gt_w / 2.0,
                gt_yc - gt_h / 2.0,
                gt_xc + gt_w / 2.0,
                gt_yc + gt_h / 2.0,
            ]
        ).view(1, 4)

        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
        centers = pred_xywh[:, :2]
        dx = (centers[:, 0] - gt_xc).abs() / (gt_w / 2.0 * float(center_radius)).clamp_min(1.0)
        dy = (centers[:, 1] - gt_yc).abs() / (gt_h / 2.0 * float(center_radius)).clamp_min(1.0)
        center_match = (dx <= 1.0) & (dy <= 1.0)
        ious = _box_iou_xyxy(pred_xyxy, gt_xyxy).view(-1)
        near = center_match | (ious >= float(iou_threshold))

        scores = image_pred[class_channel]
        if bool(near.any()):
            candidate_scores = scores[near]
        else:
            # Fall back to nearest decoded candidates so tiny boxes or unusual
            # strides still contribute a useful gradient.
            dist = dx.square() + dy.square()
            k = min(topk, int(dist.numel()))
            candidate_scores = scores[torch.topk(-dist, k=k).indices]
        best_score = candidate_scores.max()
        losses.append(F.relu(min_conf_t - best_score).pow(2) * float(loss_scale))

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
