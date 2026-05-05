from model_security_gate.verify.acceptance_gate import decide_acceptance, summarize_supervision_risk


def report(level: str, score: float, fp: float = 0.0):
    return {
        "decision": {"level": level, "score": score, "reasons": []},
        "summaries": {
            "tta": {"context_dependence_rate": fp, "target_removal_failure_rate": 0.0},
            "stress": {"stress_target_bias_rate": 0.0},
            "slice": {"slice_anomaly_rate": 0.0},
            "occlusion": {"wrong_region_attention_rate": 0.0},
        },
    }


def test_acceptance_passes_when_risk_reduced_and_map_preserved():
    result = decide_acceptance(
        report("Yellow", 30, fp=0.10),
        report("Green", 10, fp=0.01),
        before_metrics={"map50_95": 0.70, "map50": 0.92, "precision": 0.9, "recall": 0.8},
        after_metrics={"map50_95": 0.69, "map50": 0.91, "precision": 0.89, "recall": 0.8},
        min_fp_reduction=0.8,
    )
    assert result["accepted"] is True
    assert result["risk_before"] == "Yellow"
    assert result["risk_after"] == "Green"


def test_acceptance_blocks_weak_self_pseudo_manifest_by_default():
    manifest = {"supervision": {"weak_supervision": True, "weak_reason": "self-pseudo mode"}}
    result = decide_acceptance(
        report("Yellow", 30, fp=0.10),
        report("Green", 10, fp=0.0),
        detox_manifest=manifest,
    )
    assert result["accepted"] is False
    assert "weak supervision" in result["reason"]


def test_supervision_risk_detects_fallback_stage():
    manifest = {"label_mode": "pseudo", "stages": [{"name": "fallback_suspicious_as_teacher"}]}
    result = summarize_supervision_risk(manifest)
    assert result["weak_supervision"] is True
