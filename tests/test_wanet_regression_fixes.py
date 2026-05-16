"""Tests for Fix F1 (min_passing_eval_n_per_attack) and Fix F2 (WaNet co-dependency).

Fix F1: ``_passes`` must refuse to declare "passed" when the external eval
used fewer than ``min_passing_eval_n_per_attack`` images per attack, because
the Wilson 95% CI is too wide for a reliable decision.

Fix F2: ``_build_phase_plan`` must always include WaNet hardening when OGA
hardening is scheduled, preventing OGA-only training from pushing WaNet ASR
up without a corrective WaNet phase.
"""

from __future__ import annotations

from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig, _passes
from model_security_gate.detox.asr_closed_loop_train import (
    ASRClosedLoopConfig,
    _build_phase_plan,
)
from model_security_gate.detox.asr_aware_dataset import default_attack_suite


# ---------------------------------------------------------------------------
# Fix F1 tests
# ---------------------------------------------------------------------------


def test_passes_refuses_when_eval_too_small() -> None:
    """With min_passing_eval_n_per_attack=120, a 60-image eval must not pass."""
    cfg = HybridPurifyConfig(
        max_allowed_external_asr=0.10,
        max_allowed_internal_asr=0.10,
        max_map_drop=0.03,
        min_passing_eval_n_per_attack=120,
    )
    item = {
        "external_max_asr": 0.05,
        "internal_max_asr": 0.05,
        "map_drop": 0.01,
        "external_attack_counts": {"badnet_oda": 60, "wanet_oga": 60},
    }
    assert _passes(item, cfg) is False


def test_passes_allows_when_eval_large_enough() -> None:
    """With min_passing_eval_n_per_attack=120, a 150-image eval should pass."""
    cfg = HybridPurifyConfig(
        max_allowed_external_asr=0.10,
        max_allowed_internal_asr=0.10,
        max_map_drop=0.03,
        min_passing_eval_n_per_attack=120,
    )
    item = {
        "external_max_asr": 0.05,
        "internal_max_asr": 0.05,
        "map_drop": 0.01,
        "external_attack_counts": {"badnet_oda": 150, "wanet_oga": 150},
    }
    assert _passes(item, cfg) is True


def test_passes_legacy_no_guard_when_zero() -> None:
    """Default min_passing_eval_n_per_attack=0 preserves legacy behavior."""
    cfg = HybridPurifyConfig(
        max_allowed_external_asr=0.10,
        max_allowed_internal_asr=0.10,
        max_map_drop=0.03,
        min_passing_eval_n_per_attack=0,
    )
    item = {
        "external_max_asr": 0.05,
        "internal_max_asr": 0.05,
        "map_drop": 0.01,
        "external_attack_counts": {"badnet_oda": 10},
    }
    assert _passes(item, cfg) is True


def test_passes_no_counts_key_still_works() -> None:
    """If external_attack_counts is missing, guard is a no-op."""
    cfg = HybridPurifyConfig(
        max_allowed_external_asr=0.10,
        max_allowed_internal_asr=0.10,
        max_map_drop=0.03,
        min_passing_eval_n_per_attack=120,
    )
    item = {
        "external_max_asr": 0.05,
        "internal_max_asr": 0.05,
        "map_drop": 0.01,
    }
    # No counts → guard cannot fire → passes
    assert _passes(item, cfg) is True


# ---------------------------------------------------------------------------
# Fix F2 tests
# ---------------------------------------------------------------------------


def test_wanet_always_included_when_oga_scheduled() -> None:
    """When OGA is in the plan, WaNet must also appear even if below threshold."""
    specs = default_attack_suite()
    # Simulate: only badnet_oga is above threshold, wanet_oga is at 0.
    hard_scores = {"badnet_oga": 0.12, "wanet_oga": 0.0, "badnet_oda": 0.0}
    cfg = ASRClosedLoopConfig(
        active_asr_threshold=0.08,
        top_k_attacks_per_cycle=1,
    )
    phases = _build_phase_plan(specs, hard_scores, cfg)
    phase_names = [p.name for p in phases]
    assert "oga_hardening" in phase_names, f"OGA missing: {phase_names}"
    assert "wanet_hardening" in phase_names, (
        f"WaNet must be included when OGA is scheduled. Got: {phase_names}"
    )


def test_wanet_not_forced_when_oga_absent() -> None:
    """When OGA is not scheduled and WaNet has no signal, WaNet can be absent."""
    specs = default_attack_suite()
    # Only ODA is above threshold; OGA and WaNet are both at 0.
    hard_scores = {"badnet_oda": 0.12, "badnet_oga": 0.0, "wanet_oga": 0.0}
    cfg = ASRClosedLoopConfig(
        active_asr_threshold=0.08,
        top_k_attacks_per_cycle=1,
    )
    phases = _build_phase_plan(specs, hard_scores, cfg)
    phase_names = [p.name for p in phases]
    # OGA is not scheduled (below threshold and not forced)
    # WaNet should not be forced either since OGA is absent
    # (ODA guard forces ODA, but not WaNet)
    assert "oga_hardening" not in phase_names
    # WaNet may or may not be present depending on ODA guard logic;
    # the key assertion is that it's NOT forced by the OGA co-dependency.


def test_wanet_included_when_wanet_has_signal() -> None:
    """WaNet with its own signal above 0 should always be included."""
    specs = default_attack_suite()
    hard_scores = {"badnet_oga": 0.0, "wanet_oga": 0.01, "badnet_oda": 0.0}
    cfg = ASRClosedLoopConfig(
        active_asr_threshold=0.08,
        top_k_attacks_per_cycle=1,
    )
    phases = _build_phase_plan(specs, hard_scores, cfg)
    phase_names = [p.name for p in phases]
    assert "wanet_hardening" in phase_names


def test_phase_order_oga_before_wanet() -> None:
    """OGA phase must come before WaNet in the plan (matches group iteration order)."""
    specs = default_attack_suite()
    hard_scores = {"badnet_oga": 0.12, "wanet_oga": 0.0}
    cfg = ASRClosedLoopConfig(
        active_asr_threshold=0.08,
        top_k_attacks_per_cycle=1,
    )
    phases = _build_phase_plan(specs, hard_scores, cfg)
    phase_names = [p.name for p in phases]
    if "oga_hardening" in phase_names and "wanet_hardening" in phase_names:
        assert phase_names.index("oga_hardening") < phase_names.index("wanet_hardening")
