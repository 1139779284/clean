from model_security_gate.scan.risk import compute_risk_score, load_risk_config


def test_risk_config_loads_thresholds_and_weights(tmp_path):
    cfg = tmp_path / "risk.yaml"
    cfg.write_text("weights:\n  counterfactual_tta: 0.4\nthresholds:\n  green_max: 10\n  yellow_max: 30\n", encoding="utf-8")
    weights, thresholds = load_risk_config(cfg)
    assert weights.counterfactual_tta == 0.4
    assert thresholds.green_max == 10
    assert thresholds.yellow_max == 30


def test_global_false_positive_contributes_to_slice_risk():
    decision = compute_risk_score(
        {
            "provenance": {"risk": 0.0},
            "slice": {"slice_anomaly_rate": 0.0, "global_false_positive_rate": 0.40},
            "tta": {"context_dependence_rate": 0.0, "target_removal_failure_rate": 0.0},
            "stress": {"stress_target_bias_rate": 0.0},
            "occlusion": {"wrong_region_attention_rate": 0.0},
            "channel": {"top_channels": []},
        }
    )
    assert decision.score >= 20
    assert any("全局误检率" in reason for reason in decision.reasons)


def test_global_false_negative_contributes_to_slice_risk():
    decision = compute_risk_score(
        {
            "provenance": {"risk": 0.0},
            "slice": {"slice_anomaly_rate": 0.0, "global_false_negative_rate": 0.40},
            "tta": {"context_dependence_rate": 0.0, "target_removal_failure_rate": 0.0},
            "stress": {"stress_target_bias_rate": 0.0},
            "occlusion": {"wrong_region_attention_rate": 0.0},
            "channel": {"top_channels": []},
        }
    )
    assert decision.score >= 20
    assert any("漏检率" in reason for reason in decision.reasons)
