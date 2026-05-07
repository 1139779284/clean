from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig, compare_asr_matrices, _candidate_block_reasons, _candidate_improved, _hybrid_selection_score
from model_security_gate.detox.external_hard_suite import score_for_attack_name
from model_security_gate.detox.rnp import RNPConfig


def test_compare_asr_matrices_flags_badnet_oda_worse():
    before = {"suite::badnet_oda": 0.24, "suite::badnet_oga": 0.58}
    after = {"suite::badnet_oda": 0.32, "suite::badnet_oga": 0.40}
    result = compare_asr_matrices(before, after, max_worsen=0.02)
    assert result["n_worse"] == 1
    assert result["worse"][0]["attack"] == "suite::badnet_oda"


def test_hybrid_selection_penalizes_worse_single_attack():
    cfg = HybridPurifyConfig()
    clean = _hybrid_selection_score(0.25, 0.10, 0.18, 0.0, {"worse": []}, cfg)
    worse = _hybrid_selection_score(0.25, 0.10, 0.18, 0.0, {"worse": [{"attack": "suite::badnet_oda"}]}, cfg)
    assert worse > clean + cfg.worse_attack_penalty


def test_hybrid_selection_is_external_first_when_internal_regresses():
    cfg = HybridPurifyConfig()
    baseline = _hybrid_selection_score(0.25, 0.59, 0.125, 0.0, {"worse": []}, cfg)
    external_better = _hybrid_selection_score(0.20, 0.72, 0.0875, 0.0, {"worse": []}, cfg)
    assert external_better < baseline


def test_hybrid_selection_uses_selection_map_drop_for_exploration():
    cfg = HybridPurifyConfig(selection_max_map_drop=0.06)
    baseline = _hybrid_selection_score(0.25, 0.59, 0.125, 0.0, {"worse": []}, cfg)
    external_better = _hybrid_selection_score(0.20, 0.72, 0.075, 0.042, {"worse": []}, cfg)
    assert external_better < baseline


def test_candidate_improved_accepts_same_max_lower_external_mean():
    cfg = HybridPurifyConfig(min_selection_improvement=0.005, min_external_mean_improvement=0.01)
    best = {"selection_score": 0.3316, "external_max_asr": 0.20, "external_mean_asr": 0.075}
    candidate = {"selection_score": 0.3290, "external_max_asr": 0.20, "external_mean_asr": 0.0625}
    assert _candidate_improved(candidate, best, cfg)


def test_candidate_block_reasons_report_map_and_attack_failures():
    cfg = HybridPurifyConfig(max_map_drop=0.03)
    item = {
        "map_drop": 0.05,
        "asr_compare_to_baseline": {"n_worse": 1},
    }
    reasons = _candidate_block_reasons(item, cfg)
    assert "attack_worse_than_baseline" in reasons
    assert "map_drop_exceeds_threshold" in reasons


def test_candidate_block_reasons_allow_separate_selection_map_drop():
    cfg = HybridPurifyConfig(max_map_drop=0.03, selection_max_map_drop=0.06)
    item = {
        "map_drop": 0.05,
        "asr_compare_to_baseline": {"n_worse": 0},
    }
    assert _candidate_block_reasons(item, cfg) == []


def test_rnp_config_import_is_lightweight():
    cfg = RNPConfig(unlearn_steps=3, max_images=4)
    assert cfg.unlearn_steps == 3
    assert cfg.max_images == 4


def test_phase_level_selection_defaults_are_safe():
    cfg = HybridPurifyConfig()
    assert cfg.evaluate_each_phase is True
    assert cfg.rollback_bad_phase is True
    assert cfg.external_failure_replay is True
    assert cfg.external_select_phase_checkpoints is True
    assert cfg.aggressive_lambda_oda_recall > 0
    assert cfg.oda_recall_min_conf > 0
    assert cfg.oda_recall_loss_scale >= 1


def test_attack_score_matching_keeps_oga_and_oda_separate():
    scores = {
        "badnet_oga": 1.0,
        "wanet_oga": 1.0,
        "poison_benchmark::badnet_oda": 0.875,
        "poison_benchmark::blend_oga": 0.275,
    }
    assert score_for_attack_name(scores, "badnet_oda", kind="badnet_patch", goal="oda") == 0.875
    assert score_for_attack_name(scores, "blend_oga", kind="blend", goal="oga") == 0.275


def test_attack_score_matching_does_not_assign_generic_oga_to_missing_badnet_oga():
    scores = {
        "poison_benchmark::badnet_oda": 0.875,
        "poison_benchmark::wanet_oga": 0.700,
    }
    assert score_for_attack_name(scores, "badnet_oga", kind="badnet_patch", goal="oga") == 0.0
    assert score_for_attack_name(scores, "wanet_oga", kind="wanet", goal="oga") == 0.700
