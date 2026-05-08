from __future__ import annotations

import torch

from model_security_gate.detox.oda_score_calibration import (
    oda_score_calibration_loss,
    semantic_fp_region_guard_loss,
    semantic_negative_guard_loss,
)


def _batch_with_one_target() -> dict:
    return {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
    }


def _prediction(near_target_score: float, far_target_score: float = 0.05, near_other_score: float = 0.02) -> torch.Tensor:
    pred = torch.zeros((1, 6, 4), dtype=torch.float32)
    # xywh candidates in pixels. Candidate 0 is centered on GT; candidate 1 is far.
    pred[0, :4, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0])
    pred[0, :4, 1] = torch.tensor([80.0, 80.0, 15.0, 15.0])
    pred[0, :4, 2] = torch.tensor([52.0, 51.0, 18.0, 22.0])
    pred[0, :4, 3] = torch.tensor([20.0, 20.0, 10.0, 10.0])
    pred[0, 4, 0] = near_target_score
    pred[0, 4, 1] = far_target_score
    pred[0, 4, 2] = near_target_score * 0.8
    pred[0, 4, 3] = 0.01
    pred[0, 5, 0] = near_other_score
    pred[0, 5, 2] = near_other_score
    return pred


def test_score_calibration_loss_rewards_high_near_target_score() -> None:
    batch = _batch_with_one_target()
    low = oda_score_calibration_loss(_prediction(0.05), batch, [0], conf_target=0.35)
    high = oda_score_calibration_loss(_prediction(0.80), batch, [0], conf_target=0.35)

    assert high.item() < low.item()


def test_score_calibration_loss_penalizes_far_target_outranking_near_gt() -> None:
    batch = _batch_with_one_target()
    safe = oda_score_calibration_loss(_prediction(0.55, far_target_score=0.05), batch, [0], conf_target=0.35)
    unsafe = oda_score_calibration_loss(_prediction(0.55, far_target_score=0.70), batch, [0], conf_target=0.35)

    assert unsafe.item() > safe.item()


def test_score_calibration_loss_is_zero_without_target_labels() -> None:
    batch = _batch_with_one_target()
    batch["cls"] = torch.zeros((0, 1), dtype=torch.float32)
    batch["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
    batch["batch_idx"] = torch.zeros((0,), dtype=torch.float32)

    loss = oda_score_calibration_loss(_prediction(0.05), batch, [0])

    assert loss.item() == 0.0


def test_semantic_negative_guard_only_applies_to_semantic_target_absent_images() -> None:
    pred = _prediction(0.05, far_target_score=0.80).repeat(2, 1, 1)
    batch = {
        "img": torch.zeros((2, 3, 100, 100), dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "batch_idx": torch.tensor([1.0], dtype=torch.float32),
        "im_file": ["external_semantic_green_cleanlabel_negative.jpg", "external_semantic_green_cleanlabel_positive.jpg"],
    }

    loss = semantic_negative_guard_loss(pred, batch, [0], semantic_keywords=("semantic",), max_target_score=0.05)

    assert loss.item() > 0.0


def test_semantic_negative_guard_skips_nonsemantic_images() -> None:
    pred = _prediction(0.05, far_target_score=0.80)
    batch = {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "im_file": ["external_blend_oga_negative.jpg"],
    }

    loss = semantic_negative_guard_loss(pred, batch, [0], semantic_keywords=("semantic",))

    assert loss.item() == 0.0


def test_semantic_fp_region_guard_targets_matching_false_positive_region() -> None:
    pred = _prediction(0.05, far_target_score=0.80)
    batch = {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "im_file": ["external_semantic_green_cleanlabel_attack_0011_helm_021400_r00.jpg"],
    }
    regions = {"attack_0011_helm_021400": [[72.0, 72.0, 88.0, 88.0]]}

    loss = semantic_fp_region_guard_loss(pred, batch, [0], regions, max_target_score=0.03)

    assert loss.item() > 0.0


def test_semantic_fp_region_guard_skips_target_present_images() -> None:
    pred = _prediction(0.05, far_target_score=0.80)
    batch = _batch_with_one_target()
    batch["im_file"] = ["external_semantic_green_cleanlabel_attack_0011_helm_021400_r00.jpg"]
    regions = {"attack_0011_helm_021400": [[40.0, 40.0, 60.0, 60.0]]}

    loss = semantic_fp_region_guard_loss(pred, batch, [0], regions, max_target_score=0.03)

    assert loss.item() == 0.0


def test_semantic_fp_region_guard_skips_unmatched_images() -> None:
    pred = _prediction(0.05, far_target_score=0.80)
    batch = {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "im_file": ["external_semantic_green_cleanlabel_other_image_r00.jpg"],
    }
    regions = {"attack_0011_helm_021400": [[72.0, 72.0, 88.0, 88.0]]}

    loss = semantic_fp_region_guard_loss(pred, batch, [0], regions, max_target_score=0.03)

    assert loss.item() == 0.0
