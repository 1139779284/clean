from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig
from model_security_gate.detox.asr_closed_loop_train import ASRClosedLoopConfig, _build_phase_plan, _selection_score


def test_closed_loop_phase_plan_activates_oda_when_oda_external_asr_high():
    specs = [
        AttackTransformConfig("badnet_oga", kind="badnet_patch", goal="oga", poison_negative=True, poison_positive=False),
        AttackTransformConfig("badnet_oda", kind="badnet_patch", goal="oda", poison_negative=False, poison_positive=True),
        AttackTransformConfig("semantic_green_cleanlabel", kind="semantic_green", goal="semantic"),
    ]
    cfg = ASRClosedLoopConfig(active_asr_threshold=0.08, top_k_attacks_per_cycle=2, phase_epochs=1)
    phases = _build_phase_plan(specs, {"badnet_oda": 0.72, "semantic_green_cleanlabel": 0.30}, cfg)
    names = [p.name for p in phases]
    assert "oda_hardening" in names
    assert "semantic_hardening" in names
    assert names.index("oda_hardening") < names.index("semantic_hardening")
    oda = [p for p in phases if p.name == "oda_hardening"][0]
    assert oda.attack_repeat > cfg.base_attack_repeat
    assert oda.clean_repeat >= oda.attack_repeat


def test_closed_loop_phase_plan_runs_highest_asr_group_first():
    specs = [
        AttackTransformConfig("badnet_oga", kind="badnet_patch", goal="oga", poison_negative=True, poison_positive=False),
        AttackTransformConfig("badnet_oda", kind="badnet_patch", goal="oda", poison_negative=False, poison_positive=True),
        AttackTransformConfig("wanet_oga", kind="wanet", goal="oga", poison_negative=True, poison_positive=False),
    ]
    cfg = ASRClosedLoopConfig(active_asr_threshold=0.08, top_k_attacks_per_cycle=3, phase_epochs=1)
    phases = _build_phase_plan(specs, {"badnet_oga": 0.70, "badnet_oda": 0.95, "wanet_oga": 0.40}, cfg)
    assert phases[0].name == "oda_hardening"


def test_closed_loop_defaults_are_failure_only_and_rollback_guarded():
    cfg = ASRClosedLoopConfig()
    assert cfg.external_replay_failed_only is True
    assert cfg.rollback_on_external_regression is True
    assert cfg.rollback_on_no_improvement is True
    assert cfg.clean_recovery_every_cycle is False
    assert cfg.max_external_asr_regression > 0


def test_hardening_phases_preserve_external_trigger_pixels_by_default():
    specs = [
        AttackTransformConfig("badnet_oga", kind="badnet_patch", goal="oga", poison_negative=True, poison_positive=False),
        AttackTransformConfig("badnet_oda", kind="badnet_patch", goal="oda", poison_negative=False, poison_positive=True),
    ]
    cfg = ASRClosedLoopConfig(active_asr_threshold=0.08, top_k_attacks_per_cycle=2, phase_epochs=1)
    phases = _build_phase_plan(specs, {"badnet_oga": 0.90, "badnet_oda": 0.80}, cfg)
    names = [p.name for p in phases]
    assert "clean_recovery" not in names
    for phase in phases:
        if phase.name in {"oga_hardening", "oda_hardening"}:
            assert phase.train_kwargs["mosaic"] == 0.0
            assert phase.train_kwargs["mixup"] == 0.0
            assert phase.train_kwargs["erasing"] == 0.0


def test_selection_score_penalizes_external_mean_asr_regression():
    cfg = ASRClosedLoopConfig()
    same_max_low_mean = _selection_score(0.9, 0.0, 0.0, cfg, external_mean_asr=0.10)
    same_max_high_mean = _selection_score(0.9, 0.0, 0.0, cfg, external_mean_asr=0.60)
    assert same_max_high_mean > same_max_low_mean
