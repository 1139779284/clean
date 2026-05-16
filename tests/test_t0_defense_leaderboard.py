"""Tests for the T0 OD defense leaderboard."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from model_security_gate.t0.defense_leaderboard import (
    DefenseEntry,
    DefenseLeaderboardConfig,
    build_defense_leaderboard,
    evaluate_defense_entry,
    load_entries_from_manifest,
    mcnemar_exact_pvalue,
    render_defense_leaderboard_markdown,
    write_defense_leaderboard,
)


# ---------------------------------------------------------------------------
# McNemar exact two-sided p-value
# ---------------------------------------------------------------------------


def test_mcnemar_p_value_zero_discordant() -> None:
    assert mcnemar_exact_pvalue(0, 0) == pytest.approx(1.0)


def test_mcnemar_p_value_small_table_matches_binomial() -> None:
    # b=10, c=0 is the extreme case: two-sided p = 2 * (0.5)^10 = 2/1024.
    p = mcnemar_exact_pvalue(10, 0)
    assert p == pytest.approx(2.0 / 1024.0, rel=1e-9)


def test_mcnemar_p_value_symmetry() -> None:
    # Two-sided p-value should be symmetric in (b, c).
    assert mcnemar_exact_pvalue(4, 1) == pytest.approx(mcnemar_exact_pvalue(1, 4))


def test_mcnemar_p_value_clamped() -> None:
    # With b=c the discordant count splits 50/50 and p-value = 1.0.
    p = mcnemar_exact_pvalue(7, 7)
    assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


def _external_report(
    *,
    attack_to_success_mask: dict[str, list[bool]],
    suite: str = "suite",
) -> dict:
    rows = []
    per_attack = {}
    for attack, flags in attack_to_success_mask.items():
        per_attack[attack] = {"successes": sum(1 for f in flags if f), "total": len(flags)}
        for i, success in enumerate(flags):
            rows.append(
                {
                    "suite": suite,
                    "attack": attack,
                    "image_basename": f"{attack}_{i:03d}.jpg",
                    "success": bool(success),
                    "has_gt_target": False,
                    "n_gt_target": 0,
                }
            )
    asr_matrix = {attack: sum(f) / max(1, len(f)) for attack, f in attack_to_success_mask.items()}
    return {
        "summary": {
            "max_asr": max(asr_matrix.values()) if asr_matrix else 0.0,
            "mean_asr": sum(asr_matrix.values()) / max(1, len(asr_matrix)),
            "asr_matrix": asr_matrix,
            "top_attacks": [
                {"attack": attack, "asr": value, "n": len(attack_to_success_mask[attack])}
                for attack, value in asr_matrix.items()
            ],
        },
        "rows": rows,
        # Also surface counts at top-level so summarize_external_report can pick them up.
        "attack_success_counts": {k: v["successes"] for k, v in per_attack.items()},
        "attack_counts": {k: v["total"] for k, v in per_attack.items()},
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def test_evaluate_defense_marks_strong_reduction_with_mcnemar() -> None:
    # Poisoned model succeeds on 80/100 badnet samples and 20/100 wanet samples;
    # defense flips 70/80 successes and creates no new successes, clean mAP drop 1pt.
    poisoned_flags = {
        "badnet_oga_corner": [True] * 80 + [False] * 20,
        "wanet_oga": [True] * 20 + [False] * 80,
    }
    defended_flags = {
        "badnet_oga_corner": [False] * 70 + [True] * 10 + [False] * 20,
        "wanet_oga": [True] * 20 + [False] * 80,
    }
    entry = DefenseEntry(
        name="demo",
        poisoned_model_id="badnet_oga_corner_pr2000_seed1",
        defense="hybrid_purify_v4",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.29},
    )
    result = evaluate_defense_entry(entry)
    assert result["accepted"] is True
    badnet = result["per_attack"]["badnet_oga_corner"]
    assert badnet["poisoned_asr"] == pytest.approx(0.80)
    assert badnet["defended_asr"] == pytest.approx(0.10)
    assert badnet["mcnemar"]["b"] == 70
    assert badnet["mcnemar"]["c"] == 0
    assert badnet["mcnemar"]["significant"] is True
    wanet = result["per_attack"]["wanet_oga"]
    # Defense had no effect here, McNemar must not flag significant improvement.
    assert wanet["mcnemar"]["significant"] is False
    assert result["aggregate"]["n_attacks_with_paired_sig_improvement"] == 1
    assert 0.0 <= result["aggregate"]["od_der"] <= 1.0


def test_evaluate_defense_blocks_on_per_attack_regression() -> None:
    poisoned_flags = {
        "badnet_oga_corner": [True] * 20 + [False] * 80,
        "wanet_oga": [False] * 100,
    }
    defended_flags = {
        # Defense cuts badnet but introduces wanet regressions.
        "badnet_oga_corner": [True] * 5 + [False] * 95,
        "wanet_oga": [True] * 20 + [False] * 80,
    }
    entry = DefenseEntry(
        name="bad_defense",
        poisoned_model_id="badnet_oga_corner_pr2000_seed1",
        defense="regresser",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    result = evaluate_defense_entry(entry)
    assert result["accepted"] is False
    assert any("wanet_oga" in msg for msg in result["blockers"])
    assert result["per_attack"]["wanet_oga"]["regression"] is True


def test_evaluate_defense_blocks_on_clean_map_drop() -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 20 + [False] * 80}
    defended_flags = {"badnet_oga_corner": [False] * 100}
    entry = DefenseEntry(
        name="cleaning_too_hard",
        poisoned_model_id="x",
        defense="aggressive",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.50},
        clean_after={"map50_95": 0.40},  # 10pt drop, far beyond 3pt tolerance.
    )
    result = evaluate_defense_entry(entry)
    assert result["accepted"] is False
    assert any("clean mAP50-95 drop" in msg for msg in result["blockers"])


def test_evaluate_defense_handles_missing_clean_metrics() -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 20 + [False] * 80}
    defended_flags = {"badnet_oga_corner": [False] * 100}
    entry = DefenseEntry(
        name="no_clean",
        poisoned_model_id="x",
        defense="clean_missing",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
    )
    result = evaluate_defense_entry(entry)
    assert result["accepted"] is True
    assert any("clean mAP" in msg for msg in result["warnings"])
    assert result["clean"]["map50_95_drop"] is None
    # map_gain term should be 0 because we cannot verify clean preservation.
    assert result["aggregate"]["map_gain"] == pytest.approx(0.0)


def test_evaluate_defense_reports_zero_failure_upper_bound_for_perfect_defense() -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 30 + [False] * 270}
    defended_flags = {"badnet_oga_corner": [False] * 300}
    entry = DefenseEntry(
        name="perfect",
        poisoned_model_id="x",
        defense="zero_failure",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    result = evaluate_defense_entry(entry)
    zfu = result["per_attack"]["badnet_oga_corner"]["defended_zero_failure_upper_bound"]
    assert zfu is not None
    assert 0.0 < zfu < 0.02  # ~1% for n=300 at 95% confidence.


# ---------------------------------------------------------------------------
# Leaderboard construction
# ---------------------------------------------------------------------------


def test_build_defense_leaderboard_ranks_accepted_first() -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 50 + [False] * 50}
    defended_strong = {"badnet_oga_corner": [True] * 5 + [False] * 95}
    defended_regression = {
        "badnet_oga_corner": [True] * 10 + [False] * 90,
        # Introduce a new attack at high ASR -> must be ranked last.
        "wanet_oga": [True] * 40 + [False] * 60,
    }
    poisoned = _external_report(attack_to_success_mask=poisoned_flags)
    poisoned2 = _external_report(
        attack_to_success_mask={
            "badnet_oga_corner": [True] * 50 + [False] * 50,
            "wanet_oga": [False] * 100,
        }
    )
    entries = [
        DefenseEntry(
            name="regresser",
            poisoned_model_id="poisoned_A",
            defense="regresser",
            poisoned_external=poisoned2,
            defended_external=_external_report(attack_to_success_mask=defended_regression),
            clean_before={"map50_95": 0.30},
            clean_after={"map50_95": 0.30},
        ),
        DefenseEntry(
            name="strong",
            poisoned_model_id="poisoned_A",
            defense="strong",
            poisoned_external=poisoned,
            defended_external=_external_report(attack_to_success_mask=defended_strong),
            clean_before={"map50_95": 0.30},
            clean_after={"map50_95": 0.29},
        ),
    ]
    leaderboard = build_defense_leaderboard(entries)
    assert leaderboard["n_entries"] == 2
    assert leaderboard["n_accepted"] == 1
    assert leaderboard["rows"][0]["entry"]["name"] == "strong"
    assert leaderboard["rows"][1]["entry"]["name"] == "regresser"


def test_render_markdown_includes_mcnemar_and_od_der() -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 30 + [False] * 70}
    defended_flags = {"badnet_oga_corner": [False] * 100}
    entry = DefenseEntry(
        name="demo",
        poisoned_model_id="p1",
        defense="d1",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    leaderboard = build_defense_leaderboard([entry])
    md = render_defense_leaderboard_markdown(leaderboard)
    assert "OD Defense Leaderboard" in md
    assert "McNemar" in md
    assert "OD-DER" in md


def test_write_defense_leaderboard_writes_json_and_md(tmp_path: Path) -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 20 + [False] * 80}
    defended_flags = {"badnet_oga_corner": [False] * 100}
    entry = DefenseEntry(
        name="demo",
        poisoned_model_id="p",
        defense="d",
        poisoned_external=_external_report(attack_to_success_mask=poisoned_flags),
        defended_external=_external_report(attack_to_success_mask=defended_flags),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    leaderboard = build_defense_leaderboard([entry])
    json_path, md_path = write_defense_leaderboard(tmp_path, leaderboard)
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["rank"] == 1


# ---------------------------------------------------------------------------
# Manifest + CLI
# ---------------------------------------------------------------------------


def test_load_manifest_reads_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "x",
                        "poisoned_model_id": "p",
                        "defense": "d",
                        "poisoned_external": "p.json",
                        "defended_external": "d.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = load_entries_from_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].name == "x"


def test_cli_builds_leaderboard(tmp_path: Path) -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 30 + [False] * 70}
    defended_flags = {"badnet_oga_corner": [False] * 100}
    (tmp_path / "poisoned.json").write_text(
        json.dumps(_external_report(attack_to_success_mask=poisoned_flags)), encoding="utf-8"
    )
    (tmp_path / "defended.json").write_text(
        json.dumps(_external_report(attack_to_success_mask=defended_flags)), encoding="utf-8"
    )
    (tmp_path / "clean_before.json").write_text(json.dumps({"map50_95": 0.30}), encoding="utf-8")
    (tmp_path / "clean_after.json").write_text(json.dumps({"map50_95": 0.29}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "demo",
                        "poisoned_model_id": "badnet_oga_corner_pr2000_seed1",
                        "defense": "hybrid_purify_v4",
                        "poisoned_external": str(tmp_path / "poisoned.json"),
                        "defended_external": str(tmp_path / "defended.json"),
                        "clean_before": str(tmp_path / "clean_before.json"),
                        "clean_after": str(tmp_path / "clean_after.json"),
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
            "scripts/t0_defense_leaderboard.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out / "t0_defense_leaderboard.json").read_text(encoding="utf-8"))
    assert payload["n_entries"] == 1
    assert payload["rows"][0]["accepted"] is True


def test_cli_fail_on_no_accepted(tmp_path: Path) -> None:
    poisoned_flags = {"badnet_oga_corner": [True] * 20 + [False] * 80}
    # Defense introduces regression on a different attack.
    defended_flags = {
        "badnet_oga_corner": [True] * 5 + [False] * 95,
        "wanet_oga": [True] * 50 + [False] * 50,
    }
    (tmp_path / "poisoned.json").write_text(
        json.dumps(
            _external_report(
                attack_to_success_mask={
                    "badnet_oga_corner": poisoned_flags["badnet_oga_corner"],
                    "wanet_oga": [False] * 100,
                }
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "defended.json").write_text(
        json.dumps(_external_report(attack_to_success_mask=defended_flags)), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "bad",
                        "poisoned_model_id": "x",
                        "defense": "regresser",
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
            "scripts/t0_defense_leaderboard.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--fail-on-no-accepted",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
