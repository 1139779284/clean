"""Tests for the T0 OD Defense Certificate (contribution 3: evidence layer)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_security_gate.t0.defense_certificate import (
    DefenseCertificateConfig,
    build_defense_certificates,
    certify_defense_entry,
    holm_bonferroni_adjust,
    render_defense_certificates_markdown,
    write_defense_certificates,
)
from model_security_gate.t0.defense_leaderboard import DefenseEntry


# ---------------------------------------------------------------------------
# Holm-Bonferroni: core correctness
# ---------------------------------------------------------------------------


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni_adjust([]) == []


def test_holm_bonferroni_single_value_is_identity() -> None:
    assert holm_bonferroni_adjust([0.04]) == pytest.approx([0.04])


def test_holm_bonferroni_monotone_in_sorted_order() -> None:
    # Canonical example: raw [0.01, 0.02, 0.03] at k=3 ->
    # sorted adj = [0.03, 0.04, 0.03] -> step-down max -> [0.03, 0.04, 0.04]
    raw = [0.01, 0.02, 0.03]
    adj = holm_bonferroni_adjust(raw)
    # Rank 1: 3 * 0.01 = 0.03. Rank 2: 2 * 0.02 = 0.04. Rank 3: 1 * 0.03 = 0.03
    # -> step-down: 0.03, 0.04, max(0.04, 0.03) = 0.04
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.04)


def test_holm_bonferroni_preserves_input_order() -> None:
    # The smallest raw p is at index 2 -> it should receive the smallest adj p.
    raw = [0.03, 0.02, 0.01]
    adj = holm_bonferroni_adjust(raw)
    assert adj[2] <= adj[1] <= adj[0]


def test_holm_bonferroni_clamps_above_1() -> None:
    raw = [0.8, 0.9, 0.95]
    adj = holm_bonferroni_adjust(raw)
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_holm_bonferroni_controls_fwer_example() -> None:
    # A set of five attack p-values that would each look significant at 0.05
    # but the Holm step-down should keep only the strongest one, because
    # 5 * 0.03 = 0.15 > 0.05 for the smallest raw value.
    raw = [0.03, 0.04, 0.045, 0.049, 0.0499]
    adj = holm_bonferroni_adjust(raw)
    assert all(p > 0.05 for p in adj)


# ---------------------------------------------------------------------------
# Helpers: build synthetic external-hard-suite reports with known ground truth.
# ---------------------------------------------------------------------------


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
                    "has_gt_target": False,
                    "n_gt_target": 0,
                }
            )
    return {
        "summary": {
            "asr_matrix": asr_matrix,
            "max_asr": max(asr_matrix.values()) if asr_matrix else 0.0,
            "mean_asr": sum(asr_matrix.values()) / max(1, len(asr_matrix)),
            "top_attacks": [
                {"attack": a, "asr": v, "n": len(attack_to_mask[a])}
                for a, v in asr_matrix.items()
            ],
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap + certificate pipeline
# ---------------------------------------------------------------------------


def test_certify_strong_defense_certified_on_single_attack() -> None:
    # Defense flips 80 / 80 successes -> certified lower bound is well above 0.05.
    poisoned = _report({"badnet_oga_corner": [True] * 80 + [False] * 20})
    defended = _report({"badnet_oga_corner": [False] * 100})
    entry = DefenseEntry(
        name="strong_single",
        poisoned_model_id="p",
        defense="hybrid_purify_v4",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.29},
    )
    out = certify_defense_entry(entry, cfg=DefenseCertificateConfig(n_bootstrap=500))
    assert out["certified"] is True
    row = out["per_attack"]["badnet_oga_corner"]
    assert row["meets_acceptance"] is True
    assert row["acceptance_path"] == "reduction"
    assert row["meets_min_certified_reduction"] is True
    ci = row["bootstrap_ci"]
    assert ci["n"] == 100
    assert ci["mean_delta"] == pytest.approx(0.80, rel=1e-9)
    assert 0.6 <= row["certified_reduction_lower"] <= 0.8
    # Holm p is basically 0 for 80 vs 0 discordant.
    assert row["holm_adjusted_p"] < 1e-9
    assert row["holm_significant"] is True
    # Defended Wilson upper is near zero (0/100 successes).
    assert row["defended_wilson_upper"] <= 0.05


def test_certify_requires_all_attacks_to_meet_min_cmr() -> None:
    # badnet is strong (0.8 -> 0.0), but wanet reduction is marginal (0.30 -> 0.20).
    poisoned = _report(
        {
            "badnet_oga_corner": [True] * 80 + [False] * 20,
            "wanet_oga": [True] * 30 + [False] * 70,
        }
    )
    defended = _report(
        {
            "badnet_oga_corner": [False] * 100,
            # Only 10 of 30 trigger successes flipped.
            "wanet_oga": [True] * 20 + [False] * 80,
        }
    )
    entry = DefenseEntry(
        name="weak_on_wanet",
        poisoned_model_id="p",
        defense="partial",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=500,
            min_certified_reduction=0.15,
            # Force reduction-only path: 20% wanet ASR is well above a 5% cap.
            max_certified_asr=0.05,
        ),
    )
    # CMR is the worst attack's lower bound, so it ranks on wanet.
    assert out["aggregate"]["cmr_asr"] <= 0.15
    assert out["certified"] is False
    assert out["per_attack"]["wanet_oga"]["acceptance_path"] is None
    # Still produces a valid certificate, just blocks on warnings.
    assert any("min_certified_reduction" in msg for msg in out["warnings"])


def test_certify_blocks_on_regression_and_on_clean_map_drop() -> None:
    poisoned = _report({"badnet_oga_corner": [True] * 40 + [False] * 60})
    # Defense regresses: more successes after.
    defended = _report({"badnet_oga_corner": [True] * 60 + [False] * 40})
    entry_regression = DefenseEntry(
        name="regresser",
        poisoned_model_id="p",
        defense="bad",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(entry_regression, cfg=DefenseCertificateConfig(n_bootstrap=200))
    assert out["certified"] is False
    assert any("regressed" in msg for msg in out["blockers"])

    # Also block on clean mAP drop.
    poisoned2 = _report({"badnet_oga_corner": [True] * 80 + [False] * 20})
    defended2 = _report({"badnet_oga_corner": [False] * 100})
    entry_map = DefenseEntry(
        name="aggressive",
        poisoned_model_id="p",
        defense="kills_clean",
        poisoned_external=poisoned2,
        defended_external=defended2,
        clean_before={"map50_95": 0.50},
        clean_after={"map50_95": 0.40},
    )
    out_map = certify_defense_entry(entry_map, cfg=DefenseCertificateConfig(n_bootstrap=200))
    assert out_map["certified"] is False
    assert any("clean mAP50-95" in msg for msg in out_map["blockers"])


def test_certify_holm_reduces_false_discovery_across_attacks() -> None:
    # Five tiny attacks where each would look "significant" at alpha=0.05 if
    # reported independently.  Discordant counts of (b=4, c=0) per attack give
    # p ~= 0.125, which after Holm with k=5 becomes 0.625 on the min raw p.
    attacks = {f"atk_{i}": [True] * 4 + [False] * 96 for i in range(5)}
    defended_masks = {f"atk_{i}": [False] * 100 for i in range(5)}
    poisoned = _report(attacks)
    defended = _report(defended_masks)
    entry = DefenseEntry(
        name="marginal",
        poisoned_model_id="p",
        defense="small_effect",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(
        entry,
        cfg=DefenseCertificateConfig(
            n_bootstrap=200,
            min_certified_reduction=0.01,
            # Force reduction-only path: each attack has defended ASR 0% but
            # we want to exercise Holm failure rather than non-inferiority
            # acceptance.  Setting max_certified_asr=0.0 with an upper bound
            # that is strictly greater than zero ensures non-inferiority fails.
            max_certified_asr=0.0,
        ),
    )
    sig_count = sum(1 for row in out["per_attack"].values() if row.get("holm_significant"))
    assert sig_count == 0  # Holm correctly suppresses all.
    assert out["certified"] is False
    assert any("Holm" in msg for msg in out["warnings"])


def test_certify_reports_per_attack_bootstrap_ci_on_reduction() -> None:
    poisoned = _report({"badnet_oga_corner": [True] * 50 + [False] * 50})
    defended = _report({"badnet_oga_corner": [True] * 10 + [False] * 90})
    entry = DefenseEntry(
        name="ci_report",
        poisoned_model_id="p",
        defense="d",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    out = certify_defense_entry(entry, cfg=DefenseCertificateConfig(n_bootstrap=500))
    ci = out["per_attack"]["badnet_oga_corner"]["bootstrap_ci"]
    assert ci is not None
    assert ci["low"] <= ci["mean_delta"] <= ci["high"]
    assert ci["mean_delta"] == pytest.approx(0.40, abs=0.02)


def test_build_defense_certificates_ranks_by_cmr() -> None:
    # Three attacks. Defense_A: drops all to 0. Defense_B: drops two to 0 but
    # leaves WaNet higher.  Certified ranking primary is CMR, so defense A > B.
    all_attacks = ["badnet_oga_corner", "semantic_cleanlabel", "wanet_oga"]
    poisoned = _report({a: [True] * 60 + [False] * 40 for a in all_attacks})
    defended_a = _report({a: [False] * 100 for a in all_attacks})
    defended_b_masks = {
        "badnet_oga_corner": [False] * 100,
        "semantic_cleanlabel": [False] * 100,
        "wanet_oga": [True] * 30 + [False] * 70,
    }
    defended_b = _report(defended_b_masks)
    entries = [
        DefenseEntry(
            name="defense_A",
            poisoned_model_id="p",
            defense="A",
            poisoned_external=poisoned,
            defended_external=defended_a,
            clean_before={"map50_95": 0.30},
            clean_after={"map50_95": 0.29},
        ),
        DefenseEntry(
            name="defense_B",
            poisoned_model_id="p",
            defense="B",
            poisoned_external=poisoned,
            defended_external=defended_b,
            clean_before={"map50_95": 0.30},
            clean_after={"map50_95": 0.29},
        ),
    ]
    payload = build_defense_certificates(entries, cfg=DefenseCertificateConfig(n_bootstrap=300))
    assert payload["rows"][0]["entry"]["name"] == "defense_A"
    assert payload["rows"][1]["entry"]["name"] == "defense_B"
    assert payload["rows"][0]["aggregate"]["cmr_asr"] >= payload["rows"][1]["aggregate"]["cmr_asr"]


def test_write_and_render_markdown(tmp_path: Path) -> None:
    poisoned = _report({"badnet_oga_corner": [True] * 40 + [False] * 60})
    defended = _report({"badnet_oga_corner": [False] * 100})
    entry = DefenseEntry(
        name="demo",
        poisoned_model_id="p",
        defense="d",
        poisoned_external=poisoned,
        defended_external=defended,
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    payload = build_defense_certificates([entry], cfg=DefenseCertificateConfig(n_bootstrap=200))
    json_path, md_path = write_defense_certificates(tmp_path, payload)
    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Defense Certificate" in md
    assert "Certified Minimum Reduction" in md
    assert "paired bootstrap" in md.lower()


def test_certificate_deterministic_with_same_seed() -> None:
    poisoned = _report({"badnet_oga_corner": [True] * 40 + [False] * 60})
    defended = _report({"badnet_oga_corner": [True] * 8 + [False] * 92})
    entry = DefenseEntry(
        name="seeded",
        poisoned_model_id="p",
        defense="d",
        poisoned_external=poisoned,
        defended_external=defended,
    )
    out_a = certify_defense_entry(entry, cfg=DefenseCertificateConfig(n_bootstrap=300, seed=7))
    out_b = certify_defense_entry(entry, cfg=DefenseCertificateConfig(n_bootstrap=300, seed=7))
    ci_a = out_a["per_attack"]["badnet_oga_corner"]["bootstrap_ci"]
    ci_b = out_b["per_attack"]["badnet_oga_corner"]["bootstrap_ci"]
    assert ci_a == pytest.approx(ci_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_runs_on_manifest(tmp_path: Path) -> None:
    poisoned = _report({"badnet_oga_corner": [True] * 30 + [False] * 70})
    defended = _report({"badnet_oga_corner": [False] * 100})
    (tmp_path / "poisoned.json").write_text(json.dumps(poisoned), encoding="utf-8")
    (tmp_path / "defended.json").write_text(json.dumps(defended), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "demo",
                        "poisoned_model_id": "x",
                        "defense": "d",
                        "poisoned_external": str(tmp_path / "poisoned.json"),
                        "defended_external": str(tmp_path / "defended.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_defense_certificate.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--n-bootstrap",
            "200",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out / "t0_defense_certificate.json").read_text(encoding="utf-8"))
    assert payload["n_entries"] == 1
    assert payload["rows"][0]["certified"] is True


def test_cli_fail_on_no_certified(tmp_path: Path) -> None:
    # Regression case: certification must fail.
    poisoned = _report({"badnet_oga_corner": [True] * 20 + [False] * 80})
    defended = _report({"badnet_oga_corner": [True] * 50 + [False] * 50})
    (tmp_path / "p.json").write_text(json.dumps(poisoned), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps(defended), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "bad",
                        "poisoned_model_id": "x",
                        "defense": "regresser",
                        "poisoned_external": str(tmp_path / "p.json"),
                        "defended_external": str(tmp_path / "d.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_defense_certificate.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--n-bootstrap",
            "200",
            "--fail-on-no-certified",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
