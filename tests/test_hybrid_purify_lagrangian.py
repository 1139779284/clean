"""Tests for the Lagrangian wiring in Hybrid-PURIFY-OD.

These tests stay off the GPU: they exercise the pure-Python helpers
(``_normalize_metric_keys``, ``_bucket_for_attack``, ``_apply_lagrangian_weights``,
``_build_lagrangian_controller``) and verify that the configuration knobs
behave as designed.  They do not run ``run_hybrid_purify_detox_yolo`` itself.
"""

from __future__ import annotations

import pytest

from model_security_gate.detox.hybrid_purify_train import (
    HybridPurifyConfig,
    _apply_lagrangian_weights,
    _bucket_for_attack,
    _bucket_scales_from_lambdas,
    _build_lagrangian_controller,
    _normalize_metric_keys,
    _phase_feature_weights,
)


# ---------------------------------------------------------------------------
# _normalize_metric_keys
# ---------------------------------------------------------------------------


def test_normalize_metric_keys_strips_suite_prefix() -> None:
    raw = {
        "poison_benchmark::badnet_oga": 0.5,
        "poison_benchmark::wanet_oga": 0.1,
    }
    out = _normalize_metric_keys(raw)
    assert out == {"badnet_oga": 0.5, "wanet_oga": 0.1}


def test_normalize_metric_keys_merges_duplicates_by_max() -> None:
    raw = {"suite_a::badnet_oga": 0.2, "suite_b::badnet_oga": 0.6}
    assert _normalize_metric_keys(raw) == {"badnet_oga": 0.6}


def test_normalize_metric_keys_aliases_suite_suffixes_to_controller_names() -> None:
    raw = {
        "mask_bd::badnet_oga_mask_bd_v2_visible": 0.7,
        "mask_bd::blend_oga_mask_bd_v3_sig": 0.4,
    }
    out = _normalize_metric_keys(raw)
    assert out["badnet_oga_mask_bd_v2_visible"] == pytest.approx(0.7)
    assert out["badnet_oga"] == pytest.approx(0.7)
    assert out["blend_oga_mask_bd_v3_sig"] == pytest.approx(0.4)
    assert out["blend_oga"] == pytest.approx(0.4)


def test_normalize_metric_keys_drops_non_numeric() -> None:
    raw = {"badnet_oga": "nan-ish", "wanet_oga": 0.3}
    assert _normalize_metric_keys(raw) == {"wanet_oga": 0.3}


# ---------------------------------------------------------------------------
# _bucket_for_attack
# ---------------------------------------------------------------------------


def test_bucket_for_attack_known_names() -> None:
    assert _bucket_for_attack("badnet_oga") == "oga"
    assert _bucket_for_attack("badnet_oga_corner") == "oga"
    assert _bucket_for_attack("blend_oga") == "oga"
    assert _bucket_for_attack("wanet_oga") == "wanet"
    assert _bucket_for_attack("badnet_oda") == "oda"
    assert _bucket_for_attack("wanet_oda") == "oda"
    assert _bucket_for_attack("semantic_cleanlabel") == "semantic"
    assert _bucket_for_attack("semantic_green_cleanlabel") == "semantic"
    assert _bucket_for_attack("clean_map_drop") == "clean"


def test_bucket_for_attack_unknown_returns_none() -> None:
    assert _bucket_for_attack("not_a_real_attack") is None


# ---------------------------------------------------------------------------
# _bucket_scales_from_lambdas
# ---------------------------------------------------------------------------


def test_bucket_scales_scale_within_bounds() -> None:
    cfg = HybridPurifyConfig(
        lagrangian_lambda_max=6.0,
        lagrangian_base_scale=1.0,
        lagrangian_max_scale=4.0,
        lagrangian_min_scale=0.5,
    )
    lambdas = {
        "badnet_oga": 6.0,  # max lambda -> full scale
        "wanet_oga": 0.0,  # no violation
    }
    scales = _bucket_scales_from_lambdas(lambdas, cfg)
    assert scales["oga"] == pytest.approx(4.0)
    assert scales["wanet"] == pytest.approx(1.0)


def test_bucket_scales_clamped_to_min_scale() -> None:
    cfg = HybridPurifyConfig(
        lagrangian_lambda_max=6.0,
        lagrangian_base_scale=0.2,  # below min_scale; must clamp up
        lagrangian_max_scale=4.0,
        lagrangian_min_scale=0.5,
    )
    lambdas = {"badnet_oga": 0.0}
    scales = _bucket_scales_from_lambdas(lambdas, cfg)
    assert scales["oga"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _apply_lagrangian_weights
# ---------------------------------------------------------------------------


def test_apply_lagrangian_weights_no_op_when_no_lambdas() -> None:
    base = _phase_feature_weights("oga_hardening")
    out = _apply_lagrangian_weights(base, "oga_hardening", None, HybridPurifyConfig())
    assert out == base


def test_apply_lagrangian_weights_scales_only_bucket_keys() -> None:
    cfg = HybridPurifyConfig(
        lagrangian_lambda_max=6.0,
        lagrangian_base_scale=1.0,
        lagrangian_max_scale=3.0,
        lagrangian_min_scale=1.0,
    )
    base = _phase_feature_weights("oga_hardening")
    lambdas = {"badnet_oga": 6.0, "blend_oga": 6.0}
    out = _apply_lagrangian_weights(base, "oga_hardening", lambdas, cfg)
    # Bucket keys for oga are (lambda_oga_negative, lambda_proto_suppress, lambda_pgbd_paired).
    assert out["lambda_oga_negative"] == pytest.approx(base["lambda_oga_negative"] * 3.0)
    assert out["lambda_proto_suppress"] == pytest.approx(base["lambda_proto_suppress"] * 3.0)
    assert out["lambda_pgbd_paired"] == pytest.approx(base["lambda_pgbd_paired"] * 3.0)
    # Keys not in the bucket must remain unchanged.
    assert out["lambda_task"] == pytest.approx(base["lambda_task"])
    assert out["lambda_output_distill"] == pytest.approx(base["lambda_output_distill"])


def test_apply_lagrangian_weights_multiple_active_buckets_for_semantic_phase() -> None:
    cfg = HybridPurifyConfig(
        lagrangian_lambda_max=6.0,
        lagrangian_base_scale=1.0,
        lagrangian_max_scale=2.0,
        lagrangian_min_scale=1.0,
    )
    base = _phase_feature_weights("semantic_hardening")
    lambdas = {"semantic_cleanlabel": 6.0}
    out = _apply_lagrangian_weights(base, "semantic_hardening", lambdas, cfg)
    # Semantic bucket keys (lambda_pgbd_paired, lambda_proto_suppress, lambda_feature_distill).
    assert out["lambda_pgbd_paired"] == pytest.approx(base["lambda_pgbd_paired"] * 2.0)
    assert out["lambda_proto_suppress"] == pytest.approx(base["lambda_proto_suppress"] * 2.0)
    assert out["lambda_feature_distill"] == pytest.approx(base["lambda_feature_distill"] * 2.0)


# ---------------------------------------------------------------------------
# _build_lagrangian_controller
# ---------------------------------------------------------------------------


def test_build_lagrangian_controller_respects_cfg() -> None:
    cfg = HybridPurifyConfig(
        lagrangian_lambda_lr=0.4,
        lagrangian_lambda_min=0.1,
        lagrangian_lambda_max=7.5,
        lagrangian_decay=0.9,
    )
    ctl = _build_lagrangian_controller(cfg)
    assert ctl.lambda_lr == pytest.approx(0.4)
    assert ctl.lambda_min == pytest.approx(0.1)
    assert ctl.lambda_max == pytest.approx(7.5)
    assert ctl.decay == pytest.approx(0.9)
    names = [c.name for c in ctl.constraints]
    assert "badnet_oda" in names and "wanet_oga" in names and "clean_map_drop" in names


# ---------------------------------------------------------------------------
# Controller dynamics: lambda grows on violation and decays when satisfied.
# ---------------------------------------------------------------------------


def test_controller_lambda_grows_on_violation() -> None:
    """Violated attacks must end up with a larger lambda than satisfied ones.

    The controller update is ``new = old * decay + lr * violation``.  With
    decay < 1 the starting lambda can decrease in a single cycle even on a
    positive violation, but the key invariant is that a violated attack's
    lambda is *larger than the satisfied attack's lambda* at the same scale.
    """

    cfg = HybridPurifyConfig(
        use_lagrangian_controller=True,
        lagrangian_lambda_lr=0.5,
        lagrangian_lambda_max=10.0,
        lagrangian_decay=0.95,
    )
    ctl = _build_lagrangian_controller(cfg)
    # Force both starting lambdas to the same value so the trajectory
    # comparison is fair regardless of constraint weights.
    for name in ctl.lambdas:
        ctl.lambdas[name] = 1.0
    ctl.update({"wanet_oga": 0.40, "badnet_oga": 0.01, "clean_map_drop": 0.0})
    # Violated attack (wanet_oga) ends higher than satisfied attack (badnet_oga).
    assert ctl.lambdas["wanet_oga"] > ctl.lambdas["badnet_oga"]
    # Violated attack actually grew above 1.0 when both started equal.
    assert ctl.lambdas["wanet_oga"] > 1.0


def test_controller_lambda_shrinks_when_all_satisfied() -> None:
    cfg = HybridPurifyConfig(
        use_lagrangian_controller=True,
        lagrangian_lambda_lr=0.5,
        lagrangian_lambda_max=10.0,
        lagrangian_decay=0.85,
    )
    ctl = _build_lagrangian_controller(cfg)
    # Ten satisfied cycles should decay every lambda.
    initial = dict(ctl.lambdas)
    zero_metrics = {c.name: 0.0 for c in ctl.constraints}
    for _ in range(10):
        ctl.update(zero_metrics)
    for name, value in ctl.lambdas.items():
        assert value <= initial[name]


def test_controller_lambda_clamped_to_max() -> None:
    cfg = HybridPurifyConfig(
        use_lagrangian_controller=True,
        lagrangian_lambda_lr=5.0,
        lagrangian_lambda_max=4.0,
        lagrangian_decay=1.0,
    )
    ctl = _build_lagrangian_controller(cfg)
    for _ in range(5):
        ctl.update({"wanet_oga": 0.9})
    assert ctl.lambdas["wanet_oga"] <= 4.0 + 1e-9


# ---------------------------------------------------------------------------
# Integration: weights_pre_lagrangian vs weights in phase helper input shape.
# ---------------------------------------------------------------------------


def test_apply_lagrangian_weights_preserves_keys_and_is_pure() -> None:
    cfg = HybridPurifyConfig(
        use_lagrangian_controller=True,
        lagrangian_lambda_max=6.0,
    )
    base = _phase_feature_weights("oda_hardening")
    lambdas = {"badnet_oda": 6.0, "wanet_oda": 3.0}
    scaled = _apply_lagrangian_weights(base, "oda_hardening", lambdas, cfg)
    assert set(scaled.keys()) == set(base.keys())
    # Function must not mutate its input.
    assert _phase_feature_weights("oda_hardening") == base
