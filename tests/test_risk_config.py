from model_security_gate.scan.risk import load_risk_config


def test_risk_config_loads_thresholds_and_weights(tmp_path):
    cfg = tmp_path / "risk.yaml"
    cfg.write_text("weights:\n  counterfactual_tta: 0.4\nthresholds:\n  green_max: 10\n  yellow_max: 30\n", encoding="utf-8")
    weights, thresholds = load_risk_config(cfg)
    assert weights.counterfactual_tta == 0.4
    assert thresholds.green_max == 10
    assert thresholds.yellow_max == 30
