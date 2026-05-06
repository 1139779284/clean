from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig, compare_asr_matrices, _hybrid_selection_score
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


def test_rnp_config_import_is_lightweight():
    cfg = RNPConfig(unlearn_steps=3, max_images=4)
    assert cfg.unlearn_steps == 3
    assert cfg.max_images == 4


def test_phase_level_selection_defaults_are_safe():
    cfg = HybridPurifyConfig()
    assert cfg.evaluate_each_phase is True
    assert cfg.rollback_bad_phase is True
