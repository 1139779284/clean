from __future__ import annotations

from model_security_gate.detox.oda_score_calibration_repair import (
    blocked_by_hard_constraints,
    semantic_target_absent_max_conf,
)


def _external_result() -> dict:
    return {
        "summary": {
            "asr_matrix": {
                "suite::badnet_oda": 0.05,
                "suite::blend_oga": 0.0,
                "suite::semantic_green_cleanlabel": 0.05,
                "suite::wanet_oga": 0.0,
            }
        },
        "rows": [
            {
                "attack": "semantic_green_cleanlabel",
                "goal": "semantic",
                "has_gt_target": False,
                "n_gt_target": 0,
                "max_target_conf": 0.441,
            },
            {
                "attack": "semantic_green_cleanlabel",
                "goal": "semantic",
                "has_gt_target": True,
                "n_gt_target": 1,
                "max_target_conf": 0.9,
            },
        ],
    }


def test_semantic_target_absent_max_conf_ignores_positive_semantic_rows() -> None:
    assert semantic_target_absent_max_conf(_external_result(), semantic_names=("semantic",)) == 0.441


def test_hard_constraints_block_attack_asr_above_limit() -> None:
    blocked = blocked_by_hard_constraints(
        _external_result(),
        max_attack_asr={"semantic_green_cleanlabel": 0.0, "badnet_oda": 0.05},
    )

    assert any("semantic_green_cleanlabel" in item for item in blocked)
    assert not any("badnet_oda" in item for item in blocked)


def test_hard_constraints_block_semantic_fp_conf_above_limit() -> None:
    blocked = blocked_by_hard_constraints(
        _external_result(),
        semantic_fp_required_max_conf=0.25,
        semantic_names=("semantic",),
    )

    assert blocked == ["semantic_fp_conf>0.25:0.441"]

