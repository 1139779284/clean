from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from model_security_gate.detox.oda_loss_v2 import (
    _box_iou_xyxy,
    _extract_prediction,
    _near_candidate_indices,
    _score_to_negative_bce,
    _score_to_positive_bce,
    _score_to_prob,
    _target_label_indices,
    _xywh_to_xyxy_pixels,
)


def _zero_from_prediction(prediction: Any, batch: Dict[str, Any]) -> torch.Tensor:
    pred = _extract_prediction(prediction)
    if pred is not None:
        return pred.sum() * 0.0
    img = batch.get("img")
    if torch.is_tensor(img):
        return img.sum() * 0.0
    return torch.tensor(0.0)


def _select_top_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty((0,), device=values.device, dtype=torch.long)
    return torch.topk(values, k=min(max(1, int(k)), int(values.numel()))).indices




def _threshold_aware_negative_cap_loss(
    scores: torch.Tensor,
    *,
    max_target_score: float,
    negative_bce_weight: float = 1.0,
    margin_weight: float = 1.0,
    active_margin: float | None = None,
) -> torch.Tensor:
    """Suppress target scores only as much as needed to cross a score cap.

    The original semantic guards used full negative BCE on every selected score.
    That reliably removes known target-absent FPs, but it can also over-suppress
    the shared target head and regress ODA. In threshold-aware mode only scores
    inside the active band around the production cap receive gradient.
    """
    if scores.numel() == 0:
        return scores.sum() * 0.0
    probs = _score_to_prob(scores)
    active_scores = scores
    active_probs = probs
    if active_margin is not None:
        threshold = float(max_target_score) - float(active_margin)
        active = probs.detach() >= threshold
        if not bool(active.any()):
            return scores.sum() * 0.0
        active_scores = scores[active]
        active_probs = probs[active]
    loss = active_scores.sum() * 0.0
    if negative_bce_weight > 0:
        loss = loss + _score_to_negative_bce(active_scores) * float(negative_bce_weight)
    if margin_weight > 0:
        loss = loss + F.relu(active_probs.max() - float(max_target_score)).pow(2) * float(margin_weight)
    return loss


def oda_score_calibration_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    iou_threshold: float = 0.03,
    center_radius: float = 2.0,
    topk_near: int = 24,
    topk_far: int = 128,
    conf_target: float = 0.35,
    score_margin: float = 0.15,
    positive_bce_weight: float = 0.45,
    score_floor_weight: float = 1.0,
    far_margin_weight: float = 0.55,
    competing_margin_weight: float = 0.35,
    teacher_score_weight: float = 0.35,
) -> torch.Tensor:
    """Calibrate near-GT target score for ODA positive failures.

    Diagnostics showed the remaining ODA failures still have raw candidates near
    the GT target, often with good localization, but their target-class scores
    sit below the deployment confidence threshold. This loss is narrower than a
    generic recall loss:

    - it only acts on images that contain target-class GT labels;
    - it lifts target scores only for candidates geometrically near each GT;
    - it penalizes far target candidates outranking the near-GT candidate;
    - it penalizes non-target classes outranking target at the same candidates.

    The goal is score/ranking repair, not global target-class amplification.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
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

        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
        class_channel = 4 + cid

        gt_xywhn = bboxes[label_index]
        gt_xc = gt_xywhn[0] * img_w
        gt_yc = gt_xywhn[1] * img_h
        gt_w = gt_xywhn[2].clamp_min(1e-6) * img_w
        gt_h = gt_xywhn[3].clamp_min(1e-6) * img_h
        gt_xyxy = torch.stack([gt_xc - gt_w / 2.0, gt_yc - gt_h / 2.0, gt_xc + gt_w / 2.0, gt_yc + gt_h / 2.0])

        near_idx = _near_candidate_indices(
            pred_xywh,
            gt_xc,
            gt_yc,
            gt_w,
            gt_h,
            gt_xyxy,
            img_w=img_w,
            img_h=img_h,
            iou_threshold=float(iou_threshold),
            center_radius=float(center_radius),
            topk=int(topk_near),
        )
        if near_idx.numel() == 0:
            continue

        near_ious = _box_iou_xyxy(pred_xyxy[near_idx], gt_xyxy.view(1, 4)).view(-1)
        near_scores = image_pred[class_channel, near_idx]
        near_probs = _score_to_prob(near_scores)

        best_iou_local = torch.argmax(near_ious)
        best_score_local = torch.argmax(near_probs)
        top_iou_local = _select_top_indices(near_ious, int(topk_near))
        selected_local = torch.unique(torch.cat([best_iou_local.view(1), best_score_local.view(1), top_iou_local]))
        selected_idx = near_idx[selected_local]
        selected_scores = image_pred[class_channel, selected_idx]
        selected_probs = _score_to_prob(selected_scores)
        best_prob = selected_probs.max()

        one_loss = selected_scores.sum() * 0.0
        if positive_bce_weight > 0:
            one_loss = one_loss + _score_to_positive_bce(selected_scores) * float(positive_bce_weight)
        if score_floor_weight > 0:
            one_loss = one_loss + F.relu(float(conf_target) - best_prob).pow(2) * float(score_floor_weight)

        # Margin: target candidates far from all target GT should not outrank the
        # localized GT candidate. This is the OGA guard inside ODA-positive imgs.
        if far_margin_weight > 0:
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
                    iou_threshold=float(iou_threshold),
                    center_radius=float(center_radius),
                    topk=int(topk_near),
                )
                near_any_target[same_idx] = True
            far_scores = image_pred[class_channel, ~near_any_target]
            if far_scores.numel() > 0:
                far_scores = torch.topk(far_scores, k=min(int(topk_far), int(far_scores.numel()))).values
                far_prob = _score_to_prob(far_scores).max()
                one_loss = one_loss + F.relu(far_prob + float(score_margin) - best_prob).pow(2) * float(far_margin_weight)

        # Margin: at the same localized candidates, target should outrank
        # competing classes so the detection does not drift to head/person/etc.
        if competing_margin_weight > 0 and nc > 1:
            other_channels = [4 + k for k in range(nc) if k != cid]
            if other_channels:
                other_scores = image_pred[other_channels][:, selected_idx].reshape(-1)
                other_prob = _score_to_prob(other_scores).max()
                one_loss = one_loss + F.relu(other_prob + float(score_margin) - best_prob).pow(2) * float(competing_margin_weight)

        if teacher_pred is not None and image_index < teacher_pred.shape[0] and teacher_pred.shape[1] >= pred.shape[1]:
            t_image = teacher_pred[image_index]
            with torch.no_grad():
                t_scores = t_image[class_channel, selected_idx]
                teacher_floor = torch.maximum(_score_to_prob(t_scores).max(), torch.tensor(float(conf_target), device=device))
            one_loss = one_loss + F.relu(teacher_floor - best_prob).pow(2) * float(teacher_score_weight)

        losses.append(one_loss)

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def target_absent_score_guard_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    topk: int = 256,
    weight: float = 1.0,
) -> torch.Tensor:
    """Optional OGA guard for target-absent anchors.

    Kept separate from ``oda_score_calibration_loss`` so the first overfit smoke
    can run with zero OGA pressure. When enabled, this only suppresses target
    scores on images with no target labels.
    """
    pred = _extract_prediction(prediction)
    if pred is None:
        return _zero_from_prediction(prediction, batch)
    # Reuse the established implementation for target-absent negative-only
    # suppression. Import lazily to keep this module focused.
    from model_security_gate.detox.oda_loss_v2 import negative_target_candidate_suppression_loss

    return negative_target_candidate_suppression_loss(pred, batch, target_class_ids, topk=topk, weight=weight)


def semantic_negative_guard_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    semantic_keywords: Sequence[str] = ("semantic",),
    topk: int = 256,
    max_target_score: float = 0.05,
    margin_weight: float = 0.50,
    negative_bce_weight: float = 1.0,
    active_margin: float | None = None,
) -> torch.Tensor:
    """Suppress target scores on semantic target-absent guard images only.

    Score calibration can fix ODA by raising target scores near real GT boxes,
    but that can also revive semantic target-absent false positives. This guard
    is deliberately narrower than generic OGA suppression:

    - it only acts on images whose path/name contains a semantic keyword;
    - it skips any image that has a target-class label;
    - it penalizes only top target scores in those semantic-negative images.

    That keeps ODA-positive near-GT calibration separate from semantic-negative
    suppression.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_ids = [int(x) for x in target_class_ids if 0 <= int(x) < pred.shape[1] - 4]
    if not target_ids:
        return pred.sum() * 0.0
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
    files = batch.get("im_file") or []
    keywords = tuple(str(k).lower() for k in semantic_keywords if str(k).strip())
    losses: List[torch.Tensor] = []
    for image_index in range(pred.shape[0]):
        image_name = str(files[image_index]).lower() if image_index < len(files) else ""
        if keywords and not any(keyword in image_name for keyword in keywords):
            continue
        same = bidx == int(image_index)
        if bool(same.any()):
            same_cls = cls[same]
            if bool((same_cls[:, None] == target_tensor[None, :]).any()):
                continue
        target_channels = [4 + cid for cid in target_ids]
        scores = pred[image_index, target_channels, :].reshape(-1)
        if scores.numel() == 0:
            continue
        scores = torch.topk(scores, k=min(max(1, int(topk)), int(scores.numel()))).values
        loss = _threshold_aware_negative_cap_loss(
            scores,
            max_target_score=float(max_target_score),
            negative_bce_weight=float(negative_bce_weight),
            margin_weight=float(margin_weight),
            active_margin=active_margin,
        )
        losses.append(loss)
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def _lookup_fp_regions(
    fp_regions_by_image: Mapping[str, Sequence[Sequence[float]]] | None,
    image_name: str,
) -> List[Sequence[float]]:
    if not fp_regions_by_image:
        return []
    low = str(image_name).lower().replace("\\", "/")
    base = Path(low).name
    stem = Path(base).stem
    regions: List[Sequence[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for key, value in fp_regions_by_image.items():
        key_low = str(key).lower().replace("\\", "/")
        key_base = Path(key_low).name
        key_stem = Path(key_base).stem
        if (
            low == key_low
            or low.endswith(key_low)
            or base == key_base
            or (key_stem and key_stem in low)
            or (stem and stem in key_low)
        ):
            for region in value:
                if len(region) < 4:
                    continue
                signature = tuple(round(float(v), 3) for v in region[:4])
                if signature in seen:
                    continue
                seen.add(signature)
                regions.append(region)
    return regions


def semantic_fp_region_guard_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    fp_regions_by_image: Mapping[str, Sequence[Sequence[float]]] | None,
    *,
    topk: int = 64,
    max_target_score: float = 0.03,
    iou_threshold: float = 0.03,
    center_radius: float = 2.0,
    margin_weight: float = 1.0,
    negative_bce_weight: float = 1.0,
    active_margin: float | None = None,
) -> torch.Tensor:
    """Suppress target scores only around known semantic false-positive regions.

    The final residual smoke failure is a target-absent semantic image whose
    final false ``helmet`` box maps directly to high raw target candidates. A
    global semantic-negative guard can disturb ODA score calibration, so this
    loss is deliberately surgical:

    - it only acts on images with a recorded FP region;
    - it skips target-present images;
    - it suppresses target scores on candidates geometrically near that FP box.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    device = pred.device
    target_ids = [int(x) for x in target_class_ids if 0 <= int(x) < pred.shape[1] - 4]
    if not target_ids:
        return pred.sum() * 0.0
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
    files = batch.get("im_file") or []
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    target_channels = [4 + cid for cid in target_ids]
    losses: List[torch.Tensor] = []

    for image_index in range(pred.shape[0]):
        image_name = str(files[image_index]) if image_index < len(files) else ""
        regions = _lookup_fp_regions(fp_regions_by_image, image_name)
        if not regions:
            continue
        same = bidx == int(image_index)
        if bool(same.any()):
            same_cls = cls[same]
            if bool((same_cls[:, None] == target_tensor[None, :]).any()):
                continue

        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
        image_losses: List[torch.Tensor] = []
        for region in regions:
            if len(region) < 4:
                continue
            region_xyxy = torch.tensor([float(v) for v in region[:4]], device=device, dtype=pred.dtype)
            region_xyxy[[0, 2]] = region_xyxy[[0, 2]].clamp(0.0, float(img_w))
            region_xyxy[[1, 3]] = region_xyxy[[1, 3]].clamp(0.0, float(img_h))
            rw = (region_xyxy[2] - region_xyxy[0]).clamp_min(1.0)
            rh = (region_xyxy[3] - region_xyxy[1]).clamp_min(1.0)
            rx = (region_xyxy[0] + region_xyxy[2]) / 2.0
            ry = (region_xyxy[1] + region_xyxy[3]) / 2.0
            dx = (pred_xywh[:, 0] - rx).abs() / (rw / 2.0 * float(center_radius)).clamp_min(1.0)
            dy = (pred_xywh[:, 1] - ry).abs() / (rh / 2.0 * float(center_radius)).clamp_min(1.0)
            center_match = (dx <= 1.0) & (dy <= 1.0)
            ious = _box_iou_xyxy(pred_xyxy, region_xyxy.view(1, 4)).view(-1)
            candidate_idx = torch.where(center_match | (ious >= float(iou_threshold)))[0]
            if candidate_idx.numel() == 0:
                dist = dx.square() + dy.square()
                candidate_idx = torch.topk(-dist, k=min(max(1, int(topk)), int(dist.numel()))).indices
            scores = image_pred[target_channels][:, candidate_idx].reshape(-1)
            if scores.numel() == 0:
                continue
            scores = torch.topk(scores, k=min(max(1, int(topk)), int(scores.numel()))).values
            loss = _threshold_aware_negative_cap_loss(
                scores,
                max_target_score=float(max_target_score),
                negative_bce_weight=float(negative_bce_weight),
                margin_weight=float(margin_weight),
                active_margin=active_margin,
            )
            image_losses.append(loss)
        if image_losses:
            losses.append(torch.stack(image_losses).mean())

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()

def localized_target_score_floor_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    iou_threshold: float = 0.03,
    center_radius: float = 2.0,
    topk_near: int = 24,
    min_score: float = 0.25,
    teacher_margin: float = 0.02,
) -> torch.Tensor:
    """No-worse ODA anchor for target-present samples.

    For each target-class GT, prevent the repaired model from dropping below
    the stronger of a production floor and the frozen baseline teacher near the
    localized GT candidates. This anchors ODA while semantic FP suppression is
    applied elsewhere.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    if teacher_pred is None or teacher_pred.shape[1] < pred.shape[1]:
        return pred.sum() * 0.0
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    label_indices = _target_label_indices(batch, target_class_ids, device)
    if label_indices.numel() == 0 or bboxes.numel() == 0:
        return pred.sum() * 0.0
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    nc = pred.shape[1] - 4
    losses: List[torch.Tensor] = []
    for label_index in label_indices.tolist():
        image_index = int(bidx[label_index].item())
        cid = int(cls[label_index].item())
        if image_index < 0 or image_index >= pred.shape[0] or cid < 0 or cid >= nc:
            continue
        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        gt_xywhn = bboxes[label_index]
        gt_xc = gt_xywhn[0] * img_w
        gt_yc = gt_xywhn[1] * img_h
        gt_w = gt_xywhn[2].clamp_min(1e-6) * img_w
        gt_h = gt_xywhn[3].clamp_min(1e-6) * img_h
        gt_xyxy = torch.stack([gt_xc - gt_w / 2.0, gt_yc - gt_h / 2.0, gt_xc + gt_w / 2.0, gt_yc + gt_h / 2.0])
        near_idx = _near_candidate_indices(
            pred_xywh, gt_xc, gt_yc, gt_w, gt_h, gt_xyxy,
            img_w=img_w, img_h=img_h,
            iou_threshold=float(iou_threshold), center_radius=float(center_radius), topk=int(topk_near),
        )
        if near_idx.numel() == 0:
            continue
        class_channel = 4 + cid
        student_prob = _score_to_prob(image_pred[class_channel, near_idx]).max()
        with torch.no_grad():
            teacher_prob = _score_to_prob(teacher_pred[image_index, class_channel, near_idx]).max()
            floor = torch.maximum(
                torch.tensor(float(min_score), device=device, dtype=student_prob.dtype),
                (teacher_prob - float(teacher_margin)).clamp_min(0.0),
            )
        losses.append(F.relu(floor - student_prob).pow(2))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def target_absent_teacher_cap_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    topk: int = 256,
    max_target_score: float = 0.25,
    teacher_margin: float = 0.02,
) -> torch.Tensor:
    """No-worse target-absent anchor for OGA/semantic/WaNet replay samples.

    On images without target-class labels, the repaired model should not create
    new high target-class scores above the production cap or above the frozen
    baseline teacher plus margin.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_ids = [int(x) for x in target_class_ids if 0 <= int(x) < pred.shape[1] - 4]
    if not target_ids:
        return pred.sum() * 0.0
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
    target_channels = [4 + cid for cid in target_ids]
    losses: List[torch.Tensor] = []
    for image_index in range(pred.shape[0]):
        same = bidx == int(image_index)
        if bool(same.any()):
            same_cls = cls[same]
            if bool((same_cls[:, None] == target_tensor[None, :]).any()):
                continue
        scores = pred[image_index, target_channels, :].reshape(-1)
        if scores.numel() == 0:
            continue
        k = min(max(1, int(topk)), int(scores.numel()))
        selected = torch.topk(_score_to_prob(scores), k=k).values
        cap = torch.full_like(selected, float(max_target_score))
        if teacher_pred is not None and image_index < teacher_pred.shape[0] and teacher_pred.shape[1] >= pred.shape[1]:
            with torch.no_grad():
                t_scores = teacher_pred[image_index, target_channels, :].reshape(-1)
                t_probs = torch.topk(_score_to_prob(t_scores), k=min(k, int(t_scores.numel()))).values
                if t_probs.numel() == selected.numel():
                    cap = torch.minimum(cap, (t_probs + float(teacher_margin)).clamp(0.0, 1.0))
        losses.append(F.relu(selected - cap).pow(2).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()

