"""Tests for Patch C: ODA phase must prioritize recall loss over task loss.

Before Patch C, ``_phase_feature_weights("oda_hardening")`` returned
``lambda_task=1.45`` and ``lambda_oda_recall=1.0``; standard supervised
loss averaging then diluted the ODA recall floor and the defended model
let helmet confidence collapse under a trigger.  Patch C rebalances to
make ODA recall dominate the ODA phase while keeping other phases
untouched.
"""

from __future__ import annotations

from model_security_gate.detox.hybrid_purify_train import _phase_feature_weights


def test_oda_phase_recall_weight_dominates_task() -> None:
    w = _phase_feature_weights("oda_hardening")
    assert w["lambda_oda_recall"] > w["lambda_task"], (
        f"ODA recall ({w['lambda_oda_recall']}) must dominate task loss "
        f"({w['lambda_task']}) during oda_hardening"
    )
    assert w["lambda_oda_matched"] > w["lambda_proto_suppress"], (
        "Matched-ODA loss must dominate proto-suppress during ODA phase"
    )


def test_oda_phase_proto_suppress_low() -> None:
    w = _phase_feature_weights("oda_hardening")
    assert w["lambda_proto_suppress"] <= 0.1, (
        f"ODA phase must not aggressively suppress target prototype "
        f"(got {w['lambda_proto_suppress']})"
    )


def test_other_phases_unchanged_oga() -> None:
    """OGA phase branch must still match the pre-patch values exactly."""
    expected = {
        "lambda_task": 1.15,
        "lambda_adv": 0.40,
        "lambda_output_distill": 0.35,
        "lambda_feature_distill": 0.35,
        "lambda_nad": 0.45,
        "lambda_attention": 0.15,
        "lambda_prototype": 0.25,
        "lambda_proto_suppress": 0.65,
        "lambda_oda_recall": 0.0,
        "lambda_oda_matched": 0.0,
        "lambda_oga_negative": 0.75,
        "lambda_pgbd_paired": 0.45,
    }
    actual = _phase_feature_weights("oga_hardening")
    assert actual == expected, f"OGA phase weights drifted: {actual}"


def test_other_phases_unchanged_clean_recovery() -> None:
    """Clean recovery branch must still match the pre-patch values exactly."""
    expected = {
        "lambda_task": 1.0,
        "lambda_adv": 0.08,
        "lambda_output_distill": 0.75,
        "lambda_feature_distill": 0.45,
        "lambda_nad": 0.55,
        "lambda_attention": 0.15,
        "lambda_prototype": 0.25,
        "lambda_proto_suppress": 0.05,
        "lambda_oda_recall": 0.0,
        "lambda_oda_matched": 0.0,
        "lambda_oga_negative": 0.0,
        "lambda_pgbd_paired": 0.0,
    }
    actual = _phase_feature_weights("clean_recovery")
    assert actual == expected, f"clean_recovery weights drifted: {actual}"


def test_other_phases_unchanged_semantic() -> None:
    expected = {
        "lambda_task": 1.10,
        "lambda_adv": 0.45,
        "lambda_output_distill": 0.45,
        "lambda_feature_distill": 0.65,
        "lambda_nad": 0.65,
        "lambda_attention": 0.25,
        "lambda_prototype": 0.55,
        "lambda_proto_suppress": 0.45,
        "lambda_oda_recall": 0.25,
        "lambda_oda_matched": 0.35,
        "lambda_oga_negative": 0.35,
        "lambda_pgbd_paired": 0.80,
    }
    actual = _phase_feature_weights("semantic_hardening")
    assert actual == expected, f"semantic phase weights drifted: {actual}"


def test_other_phases_unchanged_wanet() -> None:
    expected = {
        "lambda_task": 1.10,
        "lambda_adv": 0.45,
        "lambda_output_distill": 0.60,
        "lambda_feature_distill": 0.70,
        "lambda_nad": 0.70,
        "lambda_attention": 0.20,
        "lambda_prototype": 0.35,
        "lambda_proto_suppress": 0.25,
        "lambda_oda_recall": 0.35,
        "lambda_oda_matched": 0.45,
        "lambda_oga_negative": 0.15,
        "lambda_pgbd_paired": 0.80,
    }
    actual = _phase_feature_weights("wanet_hardening")
    assert actual == expected, f"wanet phase weights drifted: {actual}"
