from __future__ import annotations

import torch

from model_security_gate.detox.losses import target_recall_confidence_loss


def _batch() -> dict:
    return {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
    }


def _prediction(target_score: float, far_score: float = 0.99) -> torch.Tensor:
    pred = torch.zeros((1, 6, 3), dtype=torch.float32)
    # Candidate 0 is centered inside the GT target box.
    pred[0, :, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0, target_score, 0.0])
    # Candidate 1 has high target score but is far away, so it must not satisfy
    # ODA recall near the real object.
    pred[0, :, 1] = torch.tensor([90.0, 90.0, 8.0, 8.0, far_score, 0.0])
    pred[0, :, 2] = torch.tensor([10.0, 10.0, 8.0, 8.0, 0.05, 0.0])
    return pred


def test_oda_recall_loss_penalizes_low_target_confidence_near_gt() -> None:
    loss = target_recall_confidence_loss(_prediction(0.10), _batch(), [0], min_conf=0.50)
    assert float(loss) > 0.10


def test_oda_recall_loss_ignores_far_high_confidence_false_positive() -> None:
    loss = target_recall_confidence_loss(_prediction(0.10, far_score=0.99), _batch(), [0], min_conf=0.50)
    expected = (0.50 - 0.10) ** 2
    assert abs(float(loss) - expected) < 1e-6


def test_oda_recall_loss_is_zero_when_gt_target_is_confident() -> None:
    loss = target_recall_confidence_loss(_prediction(0.80), _batch(), [0], min_conf=0.50)
    assert float(loss) == 0.0


def test_oda_recall_loss_is_zero_without_target_labels() -> None:
    batch = _batch()
    batch["cls"] = torch.tensor([[1.0]], dtype=torch.float32)
    loss = target_recall_confidence_loss(_prediction(0.10), batch, [0], min_conf=0.50)
    assert float(loss) == 0.0
