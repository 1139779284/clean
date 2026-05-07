from __future__ import annotations

import torch

from model_security_gate.detox.oda_loss_v2 import matched_candidate_oda_loss, negative_target_candidate_suppression_loss


def _batch(has_target: bool = True) -> dict:
    cls = torch.tensor([[0.0]], dtype=torch.float32) if has_target else torch.tensor([[1.0]], dtype=torch.float32)
    return {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": cls,
        "bboxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
    }


def _prediction(target_score_near: float, target_score_far: float = 0.99) -> torch.Tensor:
    pred = torch.zeros((1, 6, 3), dtype=torch.float32)
    # Channels: x,y,w,h,cls0,cls1. Candidate 0 is near GT, candidate 1 is far.
    pred[0, :, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0, target_score_near, 0.0])
    pred[0, :, 1] = torch.tensor([90.0, 90.0, 8.0, 8.0, target_score_far, 0.0])
    pred[0, :, 2] = torch.tensor([10.0, 10.0, 8.0, 8.0, 0.05, 0.0])
    return pred


def test_matched_candidate_oda_loss_penalizes_low_score_near_gt() -> None:
    loss = matched_candidate_oda_loss(_prediction(0.05), _batch(True), [0], cls_weight=1.0, box_weight=0.0)
    assert float(loss) > 0.5


def test_matched_candidate_oda_loss_ignores_far_high_confidence_for_recall() -> None:
    # Far high-confidence false positive must not satisfy ODA recall at the GT.
    loss_far = matched_candidate_oda_loss(_prediction(0.05, 0.99), _batch(True), [0], cls_weight=1.0, box_weight=0.0)
    loss_near = matched_candidate_oda_loss(_prediction(0.95, 0.00), _batch(True), [0], cls_weight=1.0, box_weight=0.0)
    assert float(loss_far) > float(loss_near)


def test_matched_candidate_oda_loss_applies_best_candidate_floor() -> None:
    low = matched_candidate_oda_loss(
        _prediction(0.35, 0.0),
        _batch(True),
        [0],
        cls_weight=0.0,
        box_weight=0.0,
        min_score=0.80,
        best_score_weight=1.0,
        best_box_weight=0.0,
        localized_margin_weight=0.0,
    )
    high = matched_candidate_oda_loss(
        _prediction(0.90, 0.0),
        _batch(True),
        [0],
        cls_weight=0.0,
        box_weight=0.0,
        min_score=0.80,
        best_score_weight=1.0,
        best_box_weight=0.0,
        localized_margin_weight=0.0,
    )
    assert float(low) > float(high)


def test_matched_candidate_oda_loss_penalizes_far_score_margin() -> None:
    far_high = matched_candidate_oda_loss(
        _prediction(0.70, 0.95),
        _batch(True),
        [0],
        cls_weight=0.0,
        box_weight=0.0,
        best_score_weight=0.0,
        best_box_weight=0.0,
        localized_margin=0.10,
        localized_margin_weight=1.0,
    )
    far_low = matched_candidate_oda_loss(
        _prediction(0.70, 0.20),
        _batch(True),
        [0],
        cls_weight=0.0,
        box_weight=0.0,
        best_score_weight=0.0,
        best_box_weight=0.0,
        localized_margin=0.10,
        localized_margin_weight=1.0,
    )
    assert float(far_high) > float(far_low)


def test_matched_candidate_oda_loss_zero_without_target_labels() -> None:
    loss = matched_candidate_oda_loss(_prediction(0.05), _batch(False), [0])
    assert float(loss) == 0.0


def test_negative_target_candidate_suppression_only_on_target_absent() -> None:
    pred = _prediction(0.90, 0.80)
    neg_loss = negative_target_candidate_suppression_loss(pred, _batch(False), [0])
    pos_loss = negative_target_candidate_suppression_loss(pred, _batch(True), [0])
    assert float(neg_loss) > 0.0
    assert float(pos_loss) == 0.0
