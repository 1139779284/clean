"""Unobserved-constraint behaviour of MultiAttackLagrangianController.

Previously, a constraint whose metric was not present in the ``metrics`` dict
was treated as infinitely violated and its lambda was pushed to ``lambda_max``.
That caused narrow external suites (only a few attacks) to silently inflate
lambdas for attacks the operator never asked about, which distorted the
Hybrid-PURIFY bucket scales in the Lagrangian ablation.

The fix: by default, a missing metric is treated as *unobserved* - lambda is
left unchanged and the trace records ``status="unobserved"``.  The legacy
hard-fail behaviour is still available when ``treat_missing_as_unobserved=False``.
"""

from __future__ import annotations

import math

import pytest

from model_security_gate.detox.multi_attack_constraints import (
    AttackConstraint,
    MultiAttackLagrangianController,
)


def _controller(**overrides) -> MultiAttackLagrangianController:
    defaults = dict(
        constraints=[
            AttackConstraint("badnet_oga", max_value=0.05, weight=4.0),
            AttackConstraint("wanet_oga", max_value=0.05, weight=6.0),
            AttackConstraint("semantic_cleanlabel", max_value=0.0, weight=8.0),
        ],
        lambda_lr=0.5,
        lambda_max=10.0,
        decay=0.95,
    )
    defaults.update(overrides)
    return MultiAttackLagrangianController(**defaults)


def test_missing_metric_keeps_lambda_unchanged_by_default() -> None:
    ctl = _controller()
    before = dict(ctl.lambdas)
    rows = ctl.update({"badnet_oga": 0.10})  # wanet_oga / semantic_cleanlabel absent
    lookup = {row["name"]: row for row in rows}
    assert lookup["wanet_oga"]["status"] == "unobserved"
    assert lookup["wanet_oga"]["value"] is None
    assert lookup["wanet_oga"]["violation"] == pytest.approx(0.0)
    assert ctl.lambdas["wanet_oga"] == pytest.approx(before["wanet_oga"])
    assert ctl.lambdas["semantic_cleanlabel"] == pytest.approx(before["semantic_cleanlabel"])
    # The observed constraint still updates.
    assert lookup["badnet_oga"]["status"] == "violated"
    assert ctl.lambdas["badnet_oga"] != before["badnet_oga"]


def test_missing_metric_legacy_hard_behaviour_still_available() -> None:
    ctl = _controller(treat_missing_as_unobserved=False, lambda_max=7.0)
    rows = ctl.update({"badnet_oga": 0.01})  # wanet / semantic missing
    lookup = {row["name"]: row for row in rows}
    assert lookup["wanet_oga"]["status"] == "missing_hard"
    assert math.isinf(lookup["wanet_oga"]["violation"])
    assert ctl.lambdas["wanet_oga"] == pytest.approx(7.0)
    assert ctl.lambdas["semantic_cleanlabel"] == pytest.approx(7.0)


def test_status_field_distinguishes_violated_satisfied_unobserved() -> None:
    ctl = _controller()
    rows = ctl.update(
        {
            "badnet_oga": 0.20,  # violated
            "wanet_oga": 0.00,  # satisfied
            # semantic_cleanlabel absent -> unobserved
        }
    )
    lookup = {row["name"]: row["status"] for row in rows}
    assert lookup == {
        "badnet_oga": "violated",
        "wanet_oga": "satisfied",
        "semantic_cleanlabel": "unobserved",
    }


def test_invalid_value_leaves_lambda_unchanged() -> None:
    ctl = _controller()
    before = ctl.lambdas["badnet_oga"]
    rows = ctl.update({"badnet_oga": "not_a_number"})
    lookup = {row["name"]: row for row in rows}
    assert lookup["badnet_oga"]["status"] == "invalid_value"
    assert ctl.lambdas["badnet_oga"] == pytest.approx(before)


def test_fallback_to_generic_metric_key_still_works() -> None:
    # Single-constraint case: a generic ``asr`` key feeds every ASR-shaped
    # constraint whose ``metric`` default is ``asr``.
    ctl = MultiAttackLagrangianController(
        constraints=[AttackConstraint("generic", max_value=0.05, weight=1.0)],
        lambda_lr=1.0,
        decay=1.0,
    )
    before = ctl.lambdas["generic"]
    ctl.update({"asr": 0.10})
    assert ctl.lambdas["generic"] > before


def test_to_dict_round_trips_config() -> None:
    ctl = _controller(lambda_lr=0.3, lambda_min=0.1, lambda_max=9.0, decay=0.9)
    dumped = ctl.to_dict()
    assert dumped["lambda_lr"] == pytest.approx(0.3)
    assert dumped["lambda_min"] == pytest.approx(0.1)
    assert dumped["lambda_max"] == pytest.approx(9.0)
    assert dumped["decay"] == pytest.approx(0.9)
    assert dumped["treat_missing_as_unobserved"] is True
    assert set(dumped["lambdas"]) == {"badnet_oga", "wanet_oga", "semantic_cleanlabel"}
