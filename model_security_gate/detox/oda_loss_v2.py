from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F



def _find_decoded_prediction(obj: Any) -> Optional[torch.Tensor]:
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


def _zero_from_batch(batch: Dict[str, Any], fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
    if fallback is not None and torch.is_tensor(fallback):
        return fallback.sum() * 0.0
    img = batch.get("img")
    if torch.is_tensor(img):
        return img.sum() * 0.0
    return torch.tensor(0.0)


def _score_to_positive_bce(score: torch.Tensor) -> torch.Tensor:
    """Positive BCE that is safe for either probabilities or logits.

    Ultralytics decoded outputs may expose class confidence-like values in
    [0, 1] or raw-ish logits depending on version/wrapper. This helper avoids
    assuming one exact representation.
    """
    if score.numel() == 0:
        return score.sum() * 0.0
    detached = score.detach()
    if float(detached.min()) >= -1e-5 and float(detached.max()) <= 1.0 + 1e-5:
        prob = score.clamp(1e-5, 1.0 - 1e-5)
        return F.binary_cross_entropy(prob, torch.ones_like(prob), reduction="mean")
    return F.binary_cross_entropy_with_logits(score, torch.ones_like(score), reduction="mean")


def _score_to_negative_bce(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score.sum() * 0.0
    detached = score.detach()
    if float(detached.min()) >= -1e-5 and float(detached.max()) <= 1.0 + 1e-5:
        prob = score.clamp(1e-5, 1.0 - 1e-5)
        return F.binary_cross_entropy(prob, torch.zeros_like(prob), reduction="mean")
    return F.binary_cross_entropy_with_logits(score, torch.zeros_like(score), reduction="mean")


def _score_to_prob(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    detached = score.detach()
    if float(detached.min()) >= -1e-5 and float(detached.max()) <= 1.0 + 1e-5:
        return score.clamp(1e-5, 1.0 - 1e-5)
    return torch.sigmoid(score)


def _extract_prediction(prediction: Any) -> Optional[torch.Tensor]:
    pred = _find_decoded_prediction(prediction)
    if pred is None:
        return None
    pred = _prediction_channels_first(pred).float()
    if pred.ndim != 3 or pred.shape[1] < 5:
        return None
    return pred


def _target_label_indices(batch: Dict[str, Any], target_class_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    if cls.numel() == 0 or not target_class_ids:
        return torch.empty((0,), device=device, dtype=torch.long)
    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long)
    return torch.where((cls[:, None] == target_ids[None, :]).any(dim=1))[0]


def _near_candidate_indices(
    pred_xywh: torch.Tensor,
    gt_xc: torch.Tensor,
    gt_yc: torch.Tensor,
    gt_w: torch.Tensor,
    gt_h: torch.Tensor,
    gt_xyxy: torch.Tensor,
    img_w: float,
    img_h: float,
    iou_threshold: float,
    center_radius: float,
    topk: int,
) -> torch.Tensor:
    pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
    centers = pred_xywh[:, :2]
    dx = (centers[:, 0] - gt_xc).abs() / (gt_w / 2.0 * float(center_radius)).clamp_min(1.0)
    dy = (centers[:, 1] - gt_yc).abs() / (gt_h / 2.0 * float(center_radius)).clamp_min(1.0)
    center_match = (dx <= 1.0) & (dy <= 1.0)
    ious = _box_iou_xyxy(pred_xyxy, gt_xyxy.view(1, 4)).view(-1)
    near = torch.where(center_match | (ious >= float(iou_threshold)))[0]
    if near.numel() > 0:
        return near
    # Fall back to nearest candidates so tiny GT boxes or unusual strides still
    # produce a useful gradient.
    dist = dx.square() + dy.square()
    k = min(max(1, int(topk)), int(dist.numel()))
    return torch.topk(-dist, k=k).indices


def matched_candidate_oda_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    iou_threshold: float = 0.05,
    center_radius: float = 1.50,
    topk: int = 24,
    cls_weight: float = 1.0,
    box_weight: float = 0.25,
    teacher_score_weight: float = 0.25,
    teacher_box_weight: float = 0.10,
    negative_other_cls_weight: float = 0.03,
    min_score: float = 0.45,
    best_score_weight: float = 0.75,
    best_box_weight: float = 0.25,
    localized_margin: float = 0.10,
    localized_margin_weight: float = 0.20,
) -> torch.Tensor:
    """ODA v2: assignment-like recall preservation for target-present boxes.

    This is stronger than ``target_recall_confidence_loss``. For every GT target
    box it finds decoded candidates near the GT region and directly encourages
    target class confidence, while also weakly aligning the score/box to the
    clean teacher prediction when available.

    It is designed for ODA/vanishing-object failures: target-present attacked
    views must still expose a high-confidence target candidate near the real box.
    The best-candidate floor is intentionally sharper than the older averaged
    recall loss, because ODA failures are usually caused by the localized target
    candidate disappearing rather than all nearby anchors being uniformly low.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5:
        return _zero_from_batch(batch)
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    label_indices = _target_label_indices(batch, target_class_ids, device)
    if label_indices.numel() == 0 or bboxes.numel() == 0:
        return pred.sum() * 0.0

    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    nc = pred.shape[1] - 4
    losses: List[torch.Tensor] = []

    for label_index in label_indices.tolist():
        image_index = int(bidx[label_index].item())
        cid = int(cls[label_index].item())
        if image_index < 0 or image_index >= pred.shape[0] or cid < 0 or cid >= nc:
            continue
        class_channel = 4 + cid
        gt_xywhn = bboxes[label_index]
        gt_xc = gt_xywhn[0] * img_w
        gt_yc = gt_xywhn[1] * img_h
        gt_w = gt_xywhn[2].clamp_min(1e-6) * img_w
        gt_h = gt_xywhn[3].clamp_min(1e-6) * img_h
        gt_xywh = torch.stack([gt_xc, gt_yc, gt_w, gt_h])
        gt_xyxy = torch.stack([gt_xc - gt_w / 2.0, gt_yc - gt_h / 2.0, gt_xc + gt_w / 2.0, gt_yc + gt_h / 2.0])

        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        idx = _near_candidate_indices(
            pred_xywh,
            gt_xc,
            gt_yc,
            gt_w,
            gt_h,
            gt_xyxy,
            img_w=img_w,
            img_h=img_h,
            iou_threshold=iou_threshold,
            center_radius=center_radius,
            topk=topk,
        )
        if idx.numel() == 0:
            continue
        target_scores = image_pred[class_channel, idx]
        # Weight candidates by geometric closeness; still backprop through all
        # nearby candidates rather than only the max.
        near_boxes = pred_xywh[idx]
        norm_delta = torch.stack(
            [
                (near_boxes[:, 0] - gt_xc) / gt_w.clamp_min(1.0),
                (near_boxes[:, 1] - gt_yc) / gt_h.clamp_min(1.0),
                torch.log((near_boxes[:, 2].clamp_min(1.0) / gt_w.clamp_min(1.0)).clamp(1e-3, 1e3)),
                torch.log((near_boxes[:, 3].clamp_min(1.0) / gt_h.clamp_min(1.0)).clamp(1e-3, 1e3)),
            ],
            dim=1,
        )
        dist = norm_delta[:, :2].pow(2).sum(dim=1)
        geom_w = torch.softmax(-dist.detach(), dim=0)
        cls_loss = (_score_to_positive_bce(target_scores) if target_scores.numel() else pred.sum() * 0.0) * float(cls_weight)
        box_loss = (F.smooth_l1_loss(near_boxes, gt_xywh.view(1, 4).expand_as(near_boxes), reduction="none").mean(dim=1) * geom_w).sum()
        box_loss = box_loss * float(box_weight) / max(img_w, img_h)
        one_loss = cls_loss + box_loss

        target_probs = _score_to_prob(target_scores)
        if target_probs.numel() > 0:
            best_pos = torch.argmax(target_probs)
            best_prob = target_probs[best_pos]
            one_loss = one_loss + F.relu(float(min_score) - best_prob).pow(2) * float(best_score_weight)
            best_box = near_boxes[best_pos]
            one_loss = one_loss + F.smooth_l1_loss(
                best_box / max(img_w, img_h),
                gt_xywh.detach() / max(img_w, img_h),
            ) * float(best_box_weight)

            if localized_margin_weight > 0:
                near_any_target = torch.zeros((pred_xywh.shape[0],), device=device, dtype=torch.bool)
                same_image_targets = label_indices[bidx[label_indices] == image_index]
                for same_label_index in same_image_targets.tolist():
                    same_box = bboxes[same_label_index]
                    sx = same_box[0] * img_w
                    sy = same_box[1] * img_h
                    sw = same_box[2].clamp_min(1e-6) * img_w
                    sh = same_box[3].clamp_min(1e-6) * img_h
                    sxyxy = torch.stack([sx - sw / 2.0, sy - sh / 2.0, sx + sw / 2.0, sy + sh / 2.0])
                    same_idx = _near_candidate_indices(
                        pred_xywh,
                        sx,
                        sy,
                        sw,
                        sh,
                        sxyxy,
                        img_w=img_w,
                        img_h=img_h,
                        iou_threshold=iou_threshold,
                        center_radius=center_radius,
                        topk=topk,
                    )
                    near_any_target[same_idx] = True
                far_scores = image_pred[class_channel, ~near_any_target]
                if far_scores.numel() > 0:
                    if far_scores.numel() > 1024:
                        far_scores = torch.topk(far_scores, k=1024).values
                    far_prob = _score_to_prob(far_scores).max()
                    one_loss = one_loss + F.relu(far_prob - best_prob + float(localized_margin)).pow(2) * float(localized_margin_weight)

        if negative_other_cls_weight > 0 and nc > 1:
            other_channels = [4 + k for k in range(nc) if k != cid]
            if other_channels:
                other_scores = image_pred[other_channels][:, idx].reshape(-1)
                one_loss = one_loss + _score_to_negative_bce(other_scores) * float(negative_other_cls_weight)

        if teacher_pred is not None and image_index < teacher_pred.shape[0] and teacher_pred.shape[1] >= pred.shape[1]:
            t_image = teacher_pred[image_index]
            t_xywh = t_image[:4].transpose(0, 1)
            t_idx = _near_candidate_indices(
                t_xywh,
                gt_xc,
                gt_yc,
                gt_w,
                gt_h,
                gt_xyxy,
                img_w=img_w,
                img_h=img_h,
                iou_threshold=iou_threshold,
                center_radius=center_radius,
                topk=topk,
            )
            if t_idx.numel() > 0:
                with torch.no_grad():
                    t_scores = t_image[class_channel, t_idx]
                    t_best = torch.argmax(t_scores)
                    t_score = t_scores[t_best].clamp(0.0, 1.0)
                    t_box = t_xywh[t_idx[t_best]]
                student_score = target_scores.max()
                # If scores are probabilities, MSE is stable; if logits, sigmoid.
                s_prob = student_score.clamp(0.0, 1.0) if float(student_score.detach()) <= 1.0 and float(student_score.detach()) >= 0.0 else torch.sigmoid(student_score)
                one_loss = one_loss + F.mse_loss(s_prob, t_score) * float(teacher_score_weight)
                # Align to teacher's localized target box when teacher found one.
                best_student_idx = idx[torch.argmax(target_scores)]
                s_box = pred_xywh[best_student_idx]
                one_loss = one_loss + F.smooth_l1_loss(s_box / max(img_w, img_h), t_box.detach() / max(img_w, img_h)) * float(teacher_box_weight)
        losses.append(one_loss)

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def negative_target_candidate_suppression_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    topk: int = 256,
    weight: float = 1.0,
) -> torch.Tensor:
    """OGA helper: only target-absent images should suppress target candidates.

    This prevents the failure mode where OGA suppression globally lowers helmet
    confidence and worsens ODA. It is intentionally skipped for images containing
    target-class labels.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_batch(batch)
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long)
    losses: List[torch.Tensor] = []
    nc = pred.shape[1] - 4
    for i in range(pred.shape[0]):
        has_target = bool(len(cls) and ((bidx == i) & (cls[:, None] == target_ids[None, :]).any(dim=1)).any())
        if has_target:
            continue
        image_pred = pred[i]
        class_indices = [4 + int(cid) for cid in target_class_ids if 0 <= int(cid) < nc]
        if not class_indices:
            continue
        scores = image_pred[class_indices].reshape(-1)
        if scores.numel() > topk:
            scores = torch.topk(scores, k=int(topk)).values
        losses.append(_score_to_negative_bce(scores) * float(weight))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
