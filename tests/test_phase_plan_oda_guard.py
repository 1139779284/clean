"""Regression test: ODA hardening must stay in the phase plan.

Dropping ODA hardening from a cycle (because baseline ODA ASR was below
``active_asr_threshold``) turned high-confidence recalled helmets into
zero-confidence misses on badnet_oda-style images after OGA/semantic
training.  The fix forces ODA hardening to remain in the plan whenever
any ODA attack signal exists, even if it was ranked below top_k.
"""

from __future__ import annotations

from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig, default_attack_suite
from model_security_gate.detox.asr_closed_loop_train import ASRClosedLoopConfig, _build_phase_plan


def _cfg(**overrides) -> ASRClosedLoopConfig:
    d = dict(
        phase_epochs=1,
        recovery_epochs=1,
        top_k_attacks_per_cycle=3,
        active_asr_threshold=0.08,
        base_clean_repeat=2,
        recovery_clean_repeat=5,
        base_attack_repeat=1,
        max_attack_repeat=5,
        adaptive_boost=3.0,
        lr0=2e-5,
        recovery_lr0=1e-5,
    )
    d.update(overrides)
    return ASRClosedLoopConfig(**d)


def test_oda_group_included_when_below_active_threshold_but_nonzero() -> None:
    # Baseline: ODA at 1.7%, well below active_asr_threshold=8%, but nonzero.
    # OGA and semantic are large; under top_k=3 without the fix, ODA would
    # be dropped.
    specs = default_attack_suite()
    hard = {
        "badnet_oda": 0.017,
        "badnet_oga": 0.17,
        "blend_oga": 0.32,
        "semantic_green_cleanlabel": 0.08,
        "wanet_oga": 0.08,
    }
    phases = _build_phase_plan(specs, hard, _cfg())
    names = [p.name for p in phases]
    assert "oda_hardening" in names, (
        "ODA hardening must remain scheduled when ODA signal > 0, even if "
        "ranked below top_k_attacks_per_cycle. Otherwise OGA training "
        "generalises trigger-suppression and regresses ODA."
    )


def test_oda_group_excluded_when_no_signal() -> None:
    specs = default_attack_suite()
    # All zero signal: phases fall back to "all groups" because hard_scores is empty-meaning.
    # With non-empty hard_scores but zero values, the old behaviour still kept ODA out.
    # We keep that case intact to avoid training phases that nothing drives.
    hard = {
        "badnet_oda": 0.0,
        "badnet_oga": 0.17,
        "blend_oga": 0.32,
    }
    phases = _build_phase_plan(specs, hard, _cfg())
    names = [p.name for p in phases]
    assert "oga_hardening" in names
    assert "oda_hardening" not in names


def test_oda_group_included_in_early_cycle_when_no_hard_history() -> None:
    # With empty hard_scores (first cycle, before external eval history is
    # captured) the legacy "fall back to all groups" branch applies and ODA
    # must be included.
    specs = default_attack_suite()
    phases = _build_phase_plan(specs, {}, _cfg())
    names = [p.name for p in phases]
    assert "oda_hardening" in names
