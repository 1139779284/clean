import pandas as pd

from model_security_gate.scan.neuron_sensitivity import summarize_channel_scan


def test_channel_scan_evaluation_marks_insufficient_data():
    df = pd.DataFrame(
        {
            "module": ["m"],
            "channel": [1],
            "score": [10.0],
            "corr_with_target_conf": [0.9],
        }
    )
    summary = summarize_channel_scan(df, n_images=3)
    assert summary["evaluation"]["status"] == "insufficient_data"
    assert summary["evaluation"]["evidence_strength"] == "weak"


def test_channel_scan_evaluation_flags_reviewable_outliers():
    df = pd.DataFrame(
        {
            "module": ["m"] * 25,
            "channel": list(range(25)),
            "detox_score": [0.01] * 24 + [1.0],
            "corr_with_target_conf": [0.0] * 24 + [0.7],
            "positive_jump_rate": [0.0] * 24 + [0.4],
        }
    )
    summary = summarize_channel_scan(df, n_images=50)
    assert summary["evaluation"]["status"] == "review"
    assert summary["evaluation"]["has_anp_evidence"] is True
    assert summary["evaluation"]["high_risk_channels"] >= 1
