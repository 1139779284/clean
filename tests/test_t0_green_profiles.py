from model_security_gate.t0.green_profiles import build_green_profile_scorecard


def test_green_profiles_split_guard_free_and_guarded_claims():
    gate = {
        "accepted": True,
        "blocked_reasons": [],
        "metrics": {
            "guard_free": {"max_asr": 0.02, "mean_asr": 0.01},
            "guarded": {"max_asr": 0.01, "mean_asr": 0.005},
            "trigger_only": {"max_asr": 0.02, "mean_asr": 0.01},
            "map50_95_drop": -0.01,
        },
        "config": {
            "max_guard_free_asr": 0.05,
            "max_guarded_asr": 0.05,
            "max_trigger_only_asr": 0.05,
            "max_clean_map50_95_drop": 0.03,
        },
    }
    scorecard = build_green_profile_scorecard(gate, {"attacks": []})
    assert "corrected_guard_free_green" in scorecard["passed_profiles"]
    assert "guarded_deployment_green" in scorecard["passed_profiles"]
    assert scorecard["contribution_split"]["model_detox_primary"] is True


def test_green_profiles_do_not_call_guarded_weight_level_detox():
    gate = {
        "accepted": False,
        "blocked_reasons": ["missing guard-free corrected external result"],
        "metrics": {"guarded": {"max_asr": 0.01, "mean_asr": 0.005}, "map50_95_drop": 0.0},
        "config": {"max_guarded_asr": 0.05},
    }
    scorecard = build_green_profile_scorecard(gate)
    assert "guarded_deployment_green" in scorecard["passed_profiles"]
    assert "corrected_guard_free_green" not in scorecard["passed_profiles"]
    assert scorecard["contribution_split"]["model_detox_primary"] is False
