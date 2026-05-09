from model_security_gate.t0.residuals import build_frontier_plan, rank_residuals


def test_rank_residuals_prioritizes_wanet():
    report = {"asr_matrix": {"badnet_oda": 0.01, "wanet_oga": 0.08, "semantic_green_cleanlabel": 0.02}}
    rows = rank_residuals(report, min_asr=0.0)
    assert rows[0]["attack"] == "wanet_oga"
    assert rows[0]["phase"] == "geometry_consistency_and_roi_stability"


def test_build_frontier_plan_merges_reports():
    plan = build_frontier_plan([
        {"asr_matrix": {"blend_oga": 0.04}},
        {"asr_matrix": {"blend_oga": 0.02, "semantic_green_cleanlabel": 0.06}},
    ])
    assert plan["n_attacks"] == 2
    assert "semantic_causal_negative_hardening" in plan["recommended_phase_order"]
