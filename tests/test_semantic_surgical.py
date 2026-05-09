from __future__ import annotations

import torch

from model_security_gate.detox.semantic_surgical import (
    oda_target_present_preservation_loss,
    parameter_l2sp_loss,
    semantic_fp_threshold_guard_loss,
    set_surgical_trainable_scope,
    target_absent_nonexpansion_loss,
    teacher_output_stability_loss,
)


def _prediction(target_score: float = 0.50, other_score: float = 0.10) -> torch.Tensor:
    pred = torch.zeros((1, 6, 4), dtype=torch.float32)
    pred[0, :4, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0])
    pred[0, :4, 1] = torch.tensor([80.0, 80.0, 15.0, 15.0])
    pred[0, :4, 2] = torch.tensor([52.0, 51.0, 18.0, 22.0])
    pred[0, :4, 3] = torch.tensor([20.0, 20.0, 10.0, 10.0])
    pred[0, 4, :] = torch.tensor([target_score, target_score * 0.8, target_score * 0.6, 0.01])
    pred[0, 5, :] = other_score
    return pred


def _negative_batch() -> dict:
    return {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "im_file": ["external_semantic_green_cleanlabel_attack_0011_helm_021400_r00.jpg"],
    }


def _positive_batch() -> dict:
    return {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "im_file": ["badnet_oda_positive.jpg"],
    }


def test_semantic_fp_threshold_guard_is_zero_below_cap() -> None:
    regions = {"attack_0011_helm_021400": [[40.0, 40.0, 60.0, 60.0]]}
    low = semantic_fp_threshold_guard_loss(_prediction(0.20), _negative_batch(), [0], regions, cap=0.245)
    high = semantic_fp_threshold_guard_loss(_prediction(0.50), _negative_batch(), [0], regions, cap=0.245)
    assert low.item() == 0.0
    assert high.item() > low.item()


def test_semantic_fp_threshold_guard_skips_target_present() -> None:
    regions = {"attack_0011_helm_021400": [[40.0, 40.0, 60.0, 60.0]]}
    batch = _positive_batch()
    batch["im_file"] = ["external_semantic_green_cleanlabel_attack_0011_helm_021400_r00.jpg"]
    loss = semantic_fp_threshold_guard_loss(_prediction(0.80), batch, [0], regions, cap=0.245)
    assert loss.item() == 0.0


def test_target_absent_nonexpansion_allows_teacher_baseline_but_penalizes_drift() -> None:
    batch = _negative_batch()
    teacher = _prediction(0.30)
    same = target_absent_nonexpansion_loss(_prediction(0.305), batch, [0], teacher_prediction=teacher, cap=0.245, teacher_slack=0.02)
    worse = target_absent_nonexpansion_loss(_prediction(0.50), batch, [0], teacher_prediction=teacher, cap=0.245, teacher_slack=0.02)
    assert same.item() == 0.0
    assert worse.item() > same.item()


def test_oda_preservation_is_preserve_only() -> None:
    batch = _positive_batch()
    teacher = _prediction(0.60)
    ok = oda_target_present_preservation_loss(_prediction(0.59), batch, [0], teacher_prediction=teacher, slack=0.02)
    bad = oda_target_present_preservation_loss(_prediction(0.20), batch, [0], teacher_prediction=teacher, slack=0.02)
    assert ok.item() == 0.0
    assert bad.item() > ok.item()


def test_teacher_stability_zero_when_predictions_identical() -> None:
    pred = _prediction(0.30)
    loss = teacher_output_stability_loss(pred, pred.clone(), _negative_batch(), [0])
    assert loss.item() == 0.0


def test_set_surgical_trainable_scope_unfreezes_only_small_tail() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sequential(torch.nn.Linear(4, 2)))
    stats = set_surgical_trainable_scope(model, scope="head_bias", last_n_modules=1)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert stats["n_trainable_tensors"] >= 1
    assert all(name.startswith("1.") for name in trainable)
    assert all(name.endswith("bias") for name in trainable)


def test_parameter_l2sp_detects_trainable_drift() -> None:
    model = torch.nn.Linear(2, 1)
    for p in model.parameters():
        p.requires_grad_(True)
    snap = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert parameter_l2sp_loss(model, snap).item() == 0.0
    with torch.no_grad():
        model.weight.add_(1.0)
    assert parameter_l2sp_loss(model, snap).item() > 0.0
