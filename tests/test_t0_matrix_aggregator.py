"""Tests for the T0 poison-matrix aggregate evidence module."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_security_gate.t0.matrix_aggregator import (
    MatrixAggregatorConfig,
    aggregate_matrix_entries,
    aggregate_matrix_summary,
    render_matrix_aggregate_markdown,
    write_matrix_aggregate,
)


def _entry(
    *,
    attack: str,
    poison_rate: float,
    seed: int,
    asr_by_attack: dict[str, float] | None = None,
    suite: str = "t0_poison_core_attack_eval",
    weights: str | None = None,
    report: str | None = None,
    max_asr: float | None = None,
    mean_asr: float | None = None,
) -> dict[str, object]:
    matrix = asr_by_attack or {attack: 0.5}
    run = f"{attack}_pr{int(round(poison_rate * 10000)):04d}_seed{seed}"
    return {
        "attack": attack,
        "run": run,
        "poison_rate": poison_rate,
        "seed": seed,
        "epochs": 5,
        "weights": weights or f"runs/fake/{run}/best.pt",
        "report": report or f"runs/fake/{run}/report.json",
        "weights_exists": True,
        "report_exists": True,
        "max_asr": float(max(matrix.values())) if max_asr is None else max_asr,
        "mean_asr": float(sum(matrix.values()) / max(1, len(matrix))) if mean_asr is None else mean_asr,
        "asr_matrix": {f"{suite}::{name}": float(value) for name, value in matrix.items()},
    }


def test_aggregate_flags_dose_response_and_strong_pass() -> None:
    entries = [
        _entry(attack="badnet_oga_corner", poison_rate=0.01, seed=1, asr_by_attack={
            "badnet_oga_corner": 0.10, "wanet_oga": 0.02, "semantic_cleanlabel": 0.03,
        }),
        _entry(attack="badnet_oga_corner", poison_rate=0.03, seed=1, asr_by_attack={
            "badnet_oga_corner": 0.40, "wanet_oga": 0.04, "semantic_cleanlabel": 0.05,
        }),
        _entry(attack="badnet_oga_corner", poison_rate=0.05, seed=1, asr_by_attack={
            "badnet_oga_corner": 0.75, "wanet_oga": 0.06, "semantic_cleanlabel": 0.06,
        }),
        _entry(attack="badnet_oga_corner", poison_rate=0.10, seed=1, asr_by_attack={
            "badnet_oga_corner": 0.90, "wanet_oga": 0.07, "semantic_cleanlabel": 0.08,
        }),
    ]
    out = aggregate_matrix_entries(entries)
    row = out["per_attack"]["badnet_oga_corner"]
    assert row["status"] == "strong"
    assert row["n_cells"] == 4
    assert row["dose_response"]["is_monotonic"] is True
    assert row["strong_pass_rate"]["successes"] == 3  # 0.40, 0.75, 0.90 >= 0.20
    bleed = row["bleed_over"]
    # Off-target stays well below intended — no bleed-over warning.
    assert bleed["max_offtarget_asr"] <= 0.1
    assert out["status"] == "passed"
    assert out["warnings"] == []


def test_aggregate_detects_non_monotonic_dose_response() -> None:
    entries = [
        _entry(attack="wanet_oga", poison_rate=0.01, seed=1, asr_by_attack={"wanet_oga": 0.30}),
        # Regression: higher poison rate has lower ASR.
        _entry(attack="wanet_oga", poison_rate=0.03, seed=1, asr_by_attack={"wanet_oga": 0.05}),
        _entry(attack="wanet_oga", poison_rate=0.05, seed=1, asr_by_attack={"wanet_oga": 0.40}),
    ]
    out = aggregate_matrix_entries(entries)
    dose = out["per_attack"]["wanet_oga"]["dose_response"]
    assert dose["is_monotonic"] is False
    assert any("non-monotonic" in msg for msg in out["warnings"])


def test_aggregate_flags_bleed_over() -> None:
    entries = [
        _entry(
            attack="badnet_oga_corner",
            poison_rate=0.05,
            seed=1,
            asr_by_attack={"badnet_oga_corner": 0.25, "semantic_cleanlabel": 0.30, "wanet_oga": 0.10},
        ),
        _entry(
            attack="badnet_oga_corner",
            poison_rate=0.10,
            seed=1,
            asr_by_attack={"badnet_oga_corner": 0.30, "semantic_cleanlabel": 0.40, "wanet_oga": 0.10},
        ),
    ]
    out = aggregate_matrix_entries(entries)
    bleed = out["per_attack"]["badnet_oga_corner"]["bleed_over"]
    assert bleed["max_offtarget_attack"] == "semantic_cleanlabel"
    assert bleed["max_offtarget_asr"] > 0.30
    assert any("bleed-over" in msg for msg in out["warnings"])


def test_aggregate_strong_pass_uses_wilson_ci() -> None:
    entries = [
        _entry(attack="semantic_cleanlabel", poison_rate=0.05, seed=seed, asr_by_attack={"semantic_cleanlabel": 0.60})
        for seed in (1, 2, 3)
    ]
    out = aggregate_matrix_entries(entries)
    row = out["per_attack"]["semantic_cleanlabel"]
    interval = row["strong_pass_rate"]
    assert interval["successes"] == 3
    assert interval["total"] == 3
    assert interval["rate"] == pytest.approx(1.0)
    # Wilson high should always be <= 1.0, low > 0 for 3/3 at 95% confidence.
    assert 0.0 < interval["low"] < interval["high"] <= 1.0
    assert row["cv_intended_asr"] == pytest.approx(0.0)


def test_aggregate_missing_matrix_falls_back_to_top_attacks() -> None:
    entries = [
        {
            "attack": "badnet_oga_corner",
            "run": "badnet_oga_corner_pr0500_seed1",
            "poison_rate": 0.05,
            "seed": 1,
            "top_attacks": [
                {"attack": "badnet_oga_corner", "asr": 0.8, "n": 100},
                {"attack": "semantic_cleanlabel", "asr": 0.05, "n": 100},
            ],
        }
    ]
    out = aggregate_matrix_entries(entries)
    row = out["per_attack"]["badnet_oga_corner"]
    assert row["max_intended_asr"] == pytest.approx(0.8)


def test_write_matrix_aggregate_creates_json_and_markdown(tmp_path: Path) -> None:
    entries = [
        _entry(attack="badnet_oga_corner", poison_rate=0.05, seed=1, asr_by_attack={
            "badnet_oga_corner": 0.70, "wanet_oga": 0.05,
        }),
        _entry(attack="wanet_oga", poison_rate=0.10, seed=1, asr_by_attack={
            "wanet_oga": 0.35, "badnet_oga_corner": 0.05,
        }),
    ]
    summary = {"entries": entries}
    aggregate = aggregate_matrix_summary(summary)
    json_path, md_path = write_matrix_aggregate(tmp_path, aggregate)
    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Per-Attack Summary" in md
    assert "Dose-Response Curves" in md
    assert "Off-Target Bleed-Over" in md
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["per_attack"]["badnet_oga_corner"]["status"] == "strong"


def test_aggregate_supports_config_customization() -> None:
    entries = [
        _entry(attack="badnet_oga_corner", poison_rate=0.01, seed=1, asr_by_attack={"badnet_oga_corner": 0.08}),
    ]
    cfg = MatrixAggregatorConfig(strong_asr_threshold=0.05, usable_asr_threshold=0.02)
    out = aggregate_matrix_entries(entries, cfg=cfg)
    assert out["per_attack"]["badnet_oga_corner"]["status"] == "strong"


def test_render_markdown_handles_empty_entries() -> None:
    md = render_matrix_aggregate_markdown(aggregate_matrix_entries([]))
    assert "T0 Poison Matrix Aggregate Evidence" in md
    assert "None" in md


def test_cli_writes_reports(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    entries = [
        _entry(attack="badnet_oga_corner", poison_rate=0.05, seed=1, asr_by_attack={"badnet_oga_corner": 0.7}),
        _entry(attack="badnet_oga_corner", poison_rate=0.10, seed=1, asr_by_attack={"badnet_oga_corner": 0.9}),
    ]
    summary.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_poison_matrix_aggregate.py",
            "--summary-json",
            str(summary),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "status=passed" in proc.stdout
    payload = json.loads((out / "t0_poison_matrix_aggregate.json").read_text(encoding="utf-8"))
    assert payload["per_attack"]["badnet_oga_corner"]["status"] == "strong"
    assert (out / "T0_POISON_MATRIX_AGGREGATE.md").exists()


def test_cli_fails_on_warnings_when_requested(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    entries = [
        _entry(attack="wanet_oga", poison_rate=0.01, seed=1, asr_by_attack={"wanet_oga": 0.30}),
        _entry(attack="wanet_oga", poison_rate=0.03, seed=1, asr_by_attack={"wanet_oga": 0.05}),
    ]
    summary.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_poison_matrix_aggregate.py",
            "--summary-json",
            str(summary),
            "--out",
            str(out),
            "--fail-on-warnings",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
