from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig
from model_security_gate.detox.asr_closed_loop_train import ASRClosedLoopConfig, _build_phase_plan


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
    oda = [p for p in phases if p.name == "oda_hardening"][0]
    assert oda.attack_repeat > cfg.base_attack_repeat
    assert oda.clean_repeat >= oda.attack_repeat
