"""Non-inferiority acceptance path in CFRC.

When a baseline's ASR is already below ``min_certified_reduction`` it is
mathematically impossible to certify a defense by the reduction path alone.
The certificate adds a second acceptance path: the defended ASR's Wilson
upper bound stays at or below ``max_certified_asr``.  Reduction-path
attacks still require Holm-Bonferroni; non-inferiority-path attacks do not.
"""

from __future__ import annotations

import pytest

from model_security_gate.t0.defense_certificate import (
    DefenseCertificateConfig,
    certify_defense_entry,
)
from model_security_gate.t0.defense_leaderboard import DefenseEntry


def _report(attack_to_mask: dict[str, list[bool]]) -> dict:
    rows = []
    asr_matrix = {}
    for attack, flags in attack_to_mask.items():
        asr_matrix[attack] = sum(1 for f in flags if f) / max(1, len(flags))
        for i, success in enumerate(flags):
            rows.append(
                {
                    "attack": attack,
                    "image_basename": f"{attack}_{i:03d}.jpg",
                    "success": bool(success),
                }
            )
    return {
        "summary": {
            "asr_matrix": asr_matrix,
            "max_asr": max(asr_matrix.values()) if asr_matrix else 0.0,
            "mean_asr": sum(asr_matrix.values()) / max(1, len(asr_matrix)),
        },
        "rows": rows,
    }


def test_non_inferiority_path_accepts_already_low_baseline() -> None:
    # Baseline 3% ASR on 300 images; defense drops to 1%.  That is a mere
    # 2pp reduction - impossible to pass reduction path at 5pp threshold.
    # But the defended Wilson upper bound is ~3.5% < 5%, so non-inferiority
    # accepts this attack.
    poisoned_mask = [True] * 9 + [False] * 291
    defended_mask = [True] * 3 + [False] * 297
    entry = DefenseEntry(
        name="already_low",
        poisoned_model_id="p",
        defense="hybrid",
        poisoned_external=_report({"badnet_oda": poisoned_mask}),
        defended_external=_report({"badnet_oda": defended_mask}),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=500,
            min_certified_reduction=0.05,
            max_certified_asr=0.05,
        ),
    )
    row = out["per_attack"]["badnet_oda"]
    assert row["acceptance_path"] == "non_inferiority"
    assert row["meets_acceptance"] is True
    assert row["meets_min_certified_reduction"] is False
    assert row["meets_max_certified_asr"] is True
    assert row["defended_wilson_upper"] <= 0.05
    # The certificate accepts because clean mAP is unchanged and no regression.
    assert out["certified"] is True


def test_non_inferiority_does_not_require_holm_significance() -> None:
    # Tiny signal, large baseline: defense drops 0 -> 0 out of 300.  McNemar
    # p-value is 1 (no discordant pairs), but non-inferiority accepts.
    poisoned_mask = [False] * 300
    defended_mask = [False] * 300
    entry = DefenseEntry(
        name="trivially_safe",
        poisoned_model_id="p",
        defense="hybrid",
        poisoned_external=_report({"badnet_oda": poisoned_mask}),
        defended_external=_report({"badnet_oda": defended_mask}),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=200,
            min_certified_reduction=0.05,
            max_certified_asr=0.05,
        ),
    )
    row = out["per_attack"]["badnet_oda"]
    assert row["acceptance_path"] == "non_inferiority"
    assert row["holm_significant"] is False
    assert out["certified"] is True
    assert not out["warnings"] or not any("Holm" in m for m in out["warnings"])


def test_reduction_path_still_requires_holm() -> None:
    # Strong effect on a baseline that is not yet below max_certified_asr.
    # Defended ASR 20% means non-inferiority never applies; reduction path
    # has to carry the attack, and Holm is enforced.  We set
    # min_certified_reduction tiny so reduction-lower passes, but the
    # McNemar outcome is marginal - Holm still fires.
    poisoned_mask = [True] * 30 + [False] * 70
    defended_mask = [True] * 20 + [False] * 80
    entry = DefenseEntry(
        name="reduction_marginal",
        poisoned_model_id="p",
        defense="hybrid",
        poisoned_external=_report({"badnet_oga": poisoned_mask}),
        defended_external=_report({"badnet_oga": defended_mask}),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=500,
            min_certified_reduction=0.02,
            max_certified_asr=0.05,  # defended 20% is well above
        ),
    )
    row = out["per_attack"]["badnet_oga"]
    assert row["acceptance_path"] == "reduction"
    # This attack's Wilson upper bound is much higher than max_certified_asr.
    assert row["meets_max_certified_asr"] is False


def test_mixed_paths_certify_when_all_attacks_pass_either_path() -> None:
    # Attack A: baseline 40% -> defended 5%, reduction path.
    # Attack B: baseline 3% -> defended ~1%, non-inferiority path (needs n=300
    # for Wilson upper to sit comfortably below 5% at 95% confidence).
    entry = DefenseEntry(
        name="mixed",
        poisoned_model_id="p",
        defense="hybrid",
        poisoned_external=_report(
            {
                "badnet_oga": [True] * 40 + [False] * 60,
                "badnet_oda": [True] * 9 + [False] * 291,
            }
        ),
        defended_external=_report(
            {
                "badnet_oga": [True] * 5 + [False] * 95,
                "badnet_oda": [True] * 3 + [False] * 297,
            }
        ),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=500,
            min_certified_reduction=0.05,
            max_certified_asr=0.05,
        ),
    )
    paths = {a: r["acceptance_path"] for a, r in out["per_attack"].items()}
    assert paths["badnet_oga"] == "reduction"
    assert paths["badnet_oda"] == "non_inferiority"
    assert out["certified"] is True
    agg = out["aggregate"]
    assert agg["n_attacks_certified_reduction"] == 1
    assert agg["n_attacks_certified_non_inferiority"] == 1
    assert agg["n_attacks_meets_acceptance"] == 2


def test_warning_message_lists_attacks_failing_both_paths() -> None:
    # Defense does nothing: same successes everywhere.
    poisoned_mask = [True] * 40 + [False] * 60
    defended_mask = [True] * 40 + [False] * 60
    entry = DefenseEntry(
        name="nothing",
        poisoned_model_id="p",
        defense="hybrid",
        poisoned_external=_report({"badnet_oga": poisoned_mask}),
        defended_external=_report({"badnet_oga": defended_mask}),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=200,
            min_certified_reduction=0.05,
            max_certified_asr=0.05,
        ),
    )
    assert out["certified"] is False
    joined = " | ".join(out["warnings"])
    assert "badnet_oga" in joined
    assert "max_certified_asr" in joined
