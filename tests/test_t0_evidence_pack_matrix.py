"""Integration tests: T0 evidence pack + matrix aggregate wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from model_security_gate.t0.evidence_gate import T0EvidenceGateConfig
from model_security_gate.t0.matrix_aggregator import MatrixAggregatorConfig
from model_security_gate.t0.report import build_t0_evidence_pack


def _guard_free_report() -> dict:
    return {
        "max_asr": 0.0,
        "mean_asr": 0.0,
        "asr_matrix": {"badnet_oda": 0.0, "wanet_oga": 0.0},
        "top_attacks": [
            {"attack": "badnet_oda", "asr": 0.0, "n": 300},
            {"attack": "wanet_oga", "asr": 0.0, "n": 300},
        ],
    }


def _matrix_entry(
    *,
    attack: str,
    poison_rate: float,
    seed: int,
    asr_by_attack: dict[str, float],
) -> dict:
    run = f"{attack}_pr{int(round(poison_rate * 10000)):04d}_seed{seed}"
    return {
        "attack": attack,
        "run": run,
        "poison_rate": poison_rate,
        "seed": seed,
        "epochs": 5,
        "weights": f"runs/fake/{run}/best.pt",
        "report": f"runs/fake/{run}/report.json",
        "weights_exists": True,
        "report_exists": True,
        "asr_matrix": {f"suite::{name}": float(value) for name, value in asr_by_attack.items()},
    }


def test_evidence_pack_embeds_matrix_aggregate(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "entries": [
                    _matrix_entry(
                        attack="badnet_oga_corner",
                        poison_rate=0.05,
                        seed=1,
                        asr_by_attack={"badnet_oga_corner": 0.70, "wanet_oga": 0.04},
                    ),
                    _matrix_entry(
                        attack="badnet_oga_corner",
                        poison_rate=0.10,
                        seed=1,
                        asr_by_attack={"badnet_oga_corner": 0.90, "wanet_oga": 0.05},
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_t0_evidence_pack(
        out_dir=tmp_path / "pack",
        guard_free_external=_guard_free_report(),
        clean_metrics_before={"map50_95": 0.50},
        clean_metrics_after={"map50_95": 0.49},
        benchmark_audit={"passed": True, "config": {"heldout_roots": ["held"]}},
        heldout_leakage={"n_overlaps": 0},
        poison_matrix_summaries=[summary],
        cfg=T0EvidenceGateConfig(max_wilson_upper_for_t0=0.05),
    )

    assert "matrix_aggregate" in payload
    matrix = payload["matrix_aggregate"]
    assert matrix["n_entries"] == 2
    assert matrix["per_attack"]["badnet_oga_corner"]["status"] == "strong"

    pack_md = (tmp_path / "pack" / "T0_EVIDENCE_PACK.md").read_text(encoding="utf-8")
    assert "Poison-Matrix Aggregate Evidence" in pack_md
    assert "badnet_oga_corner" in pack_md
    assert (tmp_path / "pack" / "T0_POISON_MATRIX_AGGREGATE.md").exists()
    pack_json = json.loads((tmp_path / "pack" / "t0_evidence_pack.json").read_text(encoding="utf-8"))
    assert "matrix_aggregate" in pack_json


def test_evidence_pack_skip_full_matrix_report(tmp_path: Path) -> None:
    summary = {
        "entries": [
            _matrix_entry(
                attack="badnet_oga_corner",
                poison_rate=0.05,
                seed=1,
                asr_by_attack={"badnet_oga_corner": 0.70},
            )
        ]
    }
    build_t0_evidence_pack(
        out_dir=tmp_path / "pack",
        guard_free_external=_guard_free_report(),
        clean_metrics_before={"map50_95": 0.50},
        clean_metrics_after={"map50_95": 0.50},
        benchmark_audit={"passed": True},
        heldout_leakage={"n_overlaps": 0},
        poison_matrix_summaries=[summary],
        write_full_matrix_report=False,
    )
    assert not (tmp_path / "pack" / "T0_POISON_MATRIX_AGGREGATE.md").exists()
    assert (tmp_path / "pack" / "T0_EVIDENCE_PACK.md").exists()


def test_evidence_pack_dedupes_multiple_summaries(tmp_path: Path) -> None:
    entry = _matrix_entry(
        attack="badnet_oga_corner",
        poison_rate=0.05,
        seed=1,
        asr_by_attack={"badnet_oga_corner": 0.70},
    )
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s1.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    # Repeated run should be deduplicated by run key.
    s2.write_text(json.dumps({"entries": [entry, _matrix_entry(
        attack="badnet_oga_corner",
        poison_rate=0.10,
        seed=1,
        asr_by_attack={"badnet_oga_corner": 0.90},
    )]}), encoding="utf-8")
    payload = build_t0_evidence_pack(
        out_dir=tmp_path / "pack",
        guard_free_external=_guard_free_report(),
        clean_metrics_before={"map50_95": 0.50},
        clean_metrics_after={"map50_95": 0.50},
        benchmark_audit={"passed": True},
        heldout_leakage={"n_overlaps": 0},
        poison_matrix_summaries=[s1, s2],
    )
    assert payload["matrix_aggregate"]["n_entries"] == 2


def test_evidence_pack_matrix_warning_surfaces_in_markdown(tmp_path: Path) -> None:
    entries = [
        _matrix_entry(
            attack="wanet_oga",
            poison_rate=0.01,
            seed=1,
            asr_by_attack={"wanet_oga": 0.30},
        ),
        _matrix_entry(
            attack="wanet_oga",
            poison_rate=0.03,
            seed=1,
            asr_by_attack={"wanet_oga": 0.05},
        ),
    ]
    build_t0_evidence_pack(
        out_dir=tmp_path / "pack",
        guard_free_external=_guard_free_report(),
        clean_metrics_before={"map50_95": 0.5},
        clean_metrics_after={"map50_95": 0.5},
        benchmark_audit={"passed": True},
        heldout_leakage={"n_overlaps": 0},
        poison_matrix_summaries=[{"entries": entries}],
        matrix_config=MatrixAggregatorConfig(dose_response_tolerance=0.01),
    )
    md = (tmp_path / "pack" / "T0_EVIDENCE_PACK.md").read_text(encoding="utf-8")
    assert "non-monotonic" in md


def test_evidence_pack_cli_with_matrix_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "entries": [
                    _matrix_entry(
                        attack="badnet_oga_corner",
                        poison_rate=0.05,
                        seed=1,
                        asr_by_attack={"badnet_oga_corner": 0.70, "wanet_oga": 0.04},
                    ),
                    _matrix_entry(
                        attack="badnet_oga_corner",
                        poison_rate=0.10,
                        seed=1,
                        asr_by_attack={"badnet_oga_corner": 0.90, "wanet_oga": 0.05},
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    gf = tmp_path / "gf.json"
    gf.write_text(json.dumps(_guard_free_report()), encoding="utf-8")
    before = tmp_path / "before.json"
    before.write_text(json.dumps({"map50_95": 0.50}), encoding="utf-8")
    after = tmp_path / "after.json"
    after.write_text(json.dumps({"map50_95": 0.49}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"passed": True, "config": {"heldout_roots": ["held"]}}), encoding="utf-8")
    leak = tmp_path / "leak.json"
    leak.write_text(json.dumps({"n_overlaps": 0}), encoding="utf-8")

    out = tmp_path / "pack"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_evidence_pack.py",
            "--out",
            str(out),
            "--guard-free-external",
            str(gf),
            "--clean-before",
            str(before),
            "--clean-after",
            str(after),
            "--benchmark-audit",
            str(audit),
            "--heldout-leakage",
            str(leak),
            "--poison-matrix-summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "matrix status=" in proc.stdout
    payload = json.loads((out / "t0_evidence_pack.json").read_text(encoding="utf-8"))
    assert payload["matrix_aggregate"]["n_entries"] == 2
