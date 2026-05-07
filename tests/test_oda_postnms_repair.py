from __future__ import annotations

from pathlib import Path

from model_security_gate.detox.oda_postnms_repair import (
    ODAPostNMSRepairConfig,
    _filter_success_rows,
    _select_attack_names,
    select_postnms_candidate,
)


def test_filter_success_rows_keeps_selected_attack_only() -> None:
    rows = [
        {"attack": "badnet_oda", "success": "true", "image": "a.jpg"},
        {"attack": "badnet_oda", "success": "false", "image": "b.jpg"},
        {"attack": "wanet_oga", "success": "true", "image": "c.jpg"},
    ]
    out = _filter_success_rows(rows, ["badnet_oda"])
    assert len(out) == 1
    assert out[0]["image"] == "a.jpg"


def test_select_attack_names_defaults_to_oda_only() -> None:
    names = ["badnet_oda", "wanet_oga", "semantic_green_cleanlabel"]
    assert _select_attack_names(names, [], goal="oda") == ["badnet_oda"]
    assert _select_attack_names(names, ["badnet_oda"], goal="oda") == ["badnet_oda"]


def test_select_postnms_candidate_rolls_back_if_no_unblocked_improvement() -> None:
    selection = select_postnms_candidate(
        [
            {"model": "bad.pt", "score": 0.05, "blocked_attacks": ["badnet_oda"]},
            {"model": "same.pt", "score": 0.20, "blocked_attacks": []},
        ],
        baseline_score=0.10,
        fallback_model="input.pt",
        require_improvement=True,
    )
    assert selection["rolled_back"] is True
    assert selection["final_model"] == "input.pt"
    assert selection["best_by_score"]["model"] == "bad.pt"


def test_select_postnms_candidate_can_choose_unblocked_improvement() -> None:
    selection = select_postnms_candidate(
        [
            {"model": "blocked.pt", "score": 0.05, "blocked_attacks": ["wanet_oga"]},
            {"model": "safe.pt", "score": 0.08, "blocked_attacks": []},
        ],
        baseline_score=0.10,
        fallback_model="input.pt",
        require_improvement=True,
    )
    assert selection["rolled_back"] is False
    assert selection["final_model"] == "safe.pt"


def test_config_defaults_are_surgical_oda_focused() -> None:
    cfg = ODAPostNMSRepairConfig(
        model="m.pt",
        data_yaml="data.yaml",
        out_dir="runs/x",
        external_roots=["bench"],
        target_classes=["helmet"],
    )
    assert cfg.lambda_oda_matched > cfg.lambda_task
    assert cfg.lambda_oga_negative == 0.0
    assert cfg.clean_anchor_images <= cfg.failure_repeat
    assert cfg.require_improvement_for_final is True
