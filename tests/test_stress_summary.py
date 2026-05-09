import pandas as pd

from model_security_gate.scan.stress_suite import summarize_stress
from model_security_gate.scan.tta_scan import summarize_tta


def test_stress_summary_tracks_vanish_and_deformation():
    df = pd.DataFrame(
        [
            {"stress_target_bias": False, "stress_target_vanish": True, "deformation_instability": True, "target_conf_inflation": -0.8, "target_conf_drop": 0.8},
            {"stress_target_bias": True, "stress_target_vanish": False, "deformation_instability": False, "target_conf_inflation": 0.4, "target_conf_drop": -0.4},
        ]
    )
    s = summarize_stress(df)
    assert s["stress_target_vanish_rate"] == 0.5
    assert s["deformation_instability_rate"] == 0.5
    assert s["max_target_conf_drop"] == 0.8


def test_tta_summary_tracks_semantic_and_color_shortcuts():
    df = pd.DataFrame(
        [
            {"variant": "hue_rotate", "conf_drop": 0.7, "context_dependence": False, "target_removal_failure": False},
            {"variant": "context_occlude", "conf_drop": 0.7, "context_dependence": True, "target_removal_failure": False},
        ]
    )
    s = summarize_tta(df)
    assert s["semantic_shortcut_rate"] > 0
    # Newer TTA summaries only count color dependency when a detection drops
    # below the operating threshold; legacy rows without confidence columns are
    # kept compatible but are not enough to prove color-trigger behavior.
    assert s["context_color_dependency_rate"] == 0.0
