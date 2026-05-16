"""Tests for Patch D distill scales.

Patch D introduces ``output_distill_scale`` and ``feature_distill_scale``
on ``HybridPurifyConfig`` so callers can keep a trusted clean teacher's
feature-level signals while discarding its possibly-noisy final-decision
outputs.  The scales multiply the phase-wise lambda values, with defaults
of 1.0 so existing behaviour is unchanged.
"""

from __future__ import annotations

import pytest

from model_security_gate.detox.hybrid_purify_train import (
    HybridPurifyConfig,
    _run_feature_purifier_phase,
)


def test_defaults_preserve_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default scales are 1.0, weights pass through unchanged."""
    import model_security_gate.detox.hybrid_purify_train as hpt

    captured = {}

    def fake_run(fcfg):
        captured["cfg"] = fcfg
        return {"best_model": str(fcfg.out_dir) + "/fake.pt", "final_model": str(fcfg.out_dir) + "/fake.pt"}

    monkeypatch.setattr(hpt, "run_strong_detox_training", fake_run)

    cfg = HybridPurifyConfig()
    # sanity: defaults
    assert cfg.output_distill_scale == 1.0
    assert cfg.feature_distill_scale == 1.0

    result = _run_feature_purifier_phase(
        model="m.pt",
        teacher_model="t.pt",
        data_yaml="d.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oda_hardening",
        cfg=cfg,
    )
    # ODA phase base: lambda_output_distill=0.45, lambda_feature_distill=0.55
    assert result["weights"]["lambda_output_distill"] == pytest.approx(0.45)
    assert result["weights"]["lambda_feature_distill"] == pytest.approx(0.55)


def test_output_distill_zero_keeps_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting output_distill_scale=0 should zero output lambda only."""
    import model_security_gate.detox.hybrid_purify_train as hpt

    monkeypatch.setattr(hpt, "run_strong_detox_training", lambda fcfg: {"best_model": "x", "final_model": "x"})

    cfg = HybridPurifyConfig(output_distill_scale=0.0, feature_distill_scale=1.0)
    result = _run_feature_purifier_phase(
        model="m.pt",
        teacher_model="t.pt",
        data_yaml="d.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oda_hardening",
        cfg=cfg,
    )
    assert result["weights"]["lambda_output_distill"] == pytest.approx(0.0)
    # ODA phase base lambda_feature_distill=0.55, unchanged
    assert result["weights"]["lambda_feature_distill"] == pytest.approx(0.55)


def test_feature_distill_scale_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """feature_distill_scale=2.0 doubles feature distill lambda."""
    import model_security_gate.detox.hybrid_purify_train as hpt

    monkeypatch.setattr(hpt, "run_strong_detox_training", lambda fcfg: {"best_model": "x", "final_model": "x"})

    cfg = HybridPurifyConfig(output_distill_scale=1.0, feature_distill_scale=2.0)
    result = _run_feature_purifier_phase(
        model="m.pt",
        teacher_model="t.pt",
        data_yaml="d.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oga_hardening",
        cfg=cfg,
    )
    # OGA phase base: lambda_feature_distill=0.35, scale=2.0 -> 0.70
    assert result["weights"]["lambda_feature_distill"] == pytest.approx(0.70)
    # output distill preserved
    assert result["weights"]["lambda_output_distill"] == pytest.approx(0.35)


def test_scales_apply_to_all_phase_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spot-check scales on wanet and clean_recovery branches."""
    import model_security_gate.detox.hybrid_purify_train as hpt

    monkeypatch.setattr(hpt, "run_strong_detox_training", lambda fcfg: {"best_model": "x", "final_model": "x"})

    cfg = HybridPurifyConfig(output_distill_scale=0.5, feature_distill_scale=0.5)
    for phase in ("wanet_hardening", "clean_recovery"):
        result = _run_feature_purifier_phase(
            model="m.pt",
            teacher_model="t.pt",
            data_yaml="d.yaml",
            out_dir="/tmp/out",
            target_ids=[0],
            phase_name=phase,
            cfg=cfg,
        )
        # Output lambda should be halved (non-zero original)
        from model_security_gate.detox.hybrid_purify_train import _phase_feature_weights
        base = _phase_feature_weights(phase)
        assert result["weights"]["lambda_output_distill"] == pytest.approx(0.5 * base["lambda_output_distill"])
        assert result["weights"]["lambda_feature_distill"] == pytest.approx(0.5 * base["lambda_feature_distill"])
