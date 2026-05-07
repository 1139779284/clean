from __future__ import annotations

import torch

from model_security_gate.detox.oda_score_calibration import oda_score_calibration_loss


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
