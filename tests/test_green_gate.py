from model_security_gate.verify.green_gate import ProductionGreenGateConfig, evaluate_production_green_gate


def _external(semantic_conf=0.10, semantic_asr=0.0):
    return {
        "summary": {
            "max_asr": max(0.05, semantic_asr),
            "mean_asr": 0.0125,
            "asr_matrix": {
                "badnet_oda": 0.05,
                "semantic_green_cleanlabel": semantic_asr,
                "blend_oga": 0.0,
                "wanet_oga": 0.0,
            },
        },
        "rows": [
            {
                "attack": "semantic_green_cleanlabel",
                "goal": "semantic",
                "has_gt_target": False,
                "n_gt_target": 0,
                "max_target_conf": semantic_conf,
            }
        ],
    }


def test_green_gate_accepts_strict_clean_candidate():
    result = evaluate_production_green_gate(
        after_report={"risk_level": "Green"},
        before_metrics={"map50_95": 0.20},
        after_metrics={"map50_95": 0.18},
        external_result=_external(),
        config=ProductionGreenGateConfig(),
    )
    assert result.accepted, result.blocked_reasons


def test_green_gate_blocks_semantic_conf_residual():
    result = evaluate_production_green_gate(
        after_report={"risk_level": "Green"},
        before_metrics={"map50_95": 0.20},
        after_metrics={"map50_95": 0.18},
        external_result=_external(semantic_conf=0.4411),
        config=ProductionGreenGateConfig(),
    )
    assert not result.accepted
    assert any("semantic target-absent max conf" in reason for reason in result.blocked_reasons)


def test_green_gate_blocks_semantic_asr_residual_even_when_max_asr_allowed():
    result = evaluate_production_green_gate(
        after_report={"risk_level": "Green"},
        before_metrics={"map50_95": 0.20},
        after_metrics={"map50_95": 0.18},
        external_result=_external(semantic_conf=0.10, semantic_asr=0.05),
        config=ProductionGreenGateConfig(),
    )
    assert not result.accepted
    assert any("semantic_green_cleanlabel ASR" in reason for reason in result.blocked_reasons)
