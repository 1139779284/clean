"""Tests for the T0 train-eval leakage audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_security_gate.t0.leakage_audit import (
    audit_cfrc_manifest,
    audit_hybrid_manifest_against_eval,
)


def _manifest(replay: list[str], evalr: list[str], datasets: list[str]) -> dict:
    return {
        "config": {
            "external_replay_roots": list(replay),
            "external_eval_roots": list(evalr),
        },
        "external_replay_datasets": [{"attack": d} for d in datasets],
    }


def _report(attack_names: list[str]) -> dict:
    return {
        "summary": {
            "asr_matrix": {f"suite::{a}": 0.1 for a in attack_names},
        }
    }


def test_disjoint_roots_are_ok() -> None:
    # Disjoint roots AND disjoint attack families.
    manifest = _manifest(
        replay=["D:/data/replay_suite"],
        evalr=["D:/data/heldout_suite"],
        datasets=["some_replay_only_attack"],
    )
    report = _report(["badnet_oga", "wanet_oga"])
    audit = audit_hybrid_manifest_against_eval(
        hybrid_manifest=manifest,
        defended_external_report=report,
        poisoned_external_report=report,
    )
    assert audit.severity == "ok"
    assert audit.train_eval_same_roots == []
    assert audit.shared_attack_keys == []


def test_same_root_and_same_attack_is_blocked() -> None:
    manifest = _manifest(
        replay=["D:/data/suite"],
        evalr=["D:/data/suite"],
        datasets=["badnet_oga", "wanet_oga"],
    )
    report = _report(["badnet_oga", "wanet_oga"])
    audit = audit_hybrid_manifest_against_eval(
        hybrid_manifest=manifest,
        defended_external_report=report,
        poisoned_external_report=report,
    )
    assert audit.severity == "blocked"
    assert "D:/data/suite" in audit.train_eval_same_roots
    assert set(audit.shared_attack_keys) == {"badnet_oga", "wanet_oga"}


def test_same_root_but_no_declared_attack_overlap_warns() -> None:
    # Manifest lists no replay datasets but roots still match.
    manifest = _manifest(
        replay=["D:/data/suite"],
        evalr=["D:/data/suite"],
        datasets=[],
    )
    report = _report(["badnet_oga"])
    audit = audit_hybrid_manifest_against_eval(
        hybrid_manifest=manifest,
        defended_external_report=report,
        poisoned_external_report=None,
    )
    assert audit.severity == "warn"
    assert "D:/data/suite" in audit.train_eval_same_roots


def test_different_roots_same_attack_family_warns() -> None:
    manifest = _manifest(
        replay=["D:/data/suite_a"],
        evalr=["D:/data/suite_b"],
        datasets=["badnet_oga"],
    )
    report = _report(["badnet_oga"])
    audit = audit_hybrid_manifest_against_eval(
        hybrid_manifest=manifest,
        defended_external_report=report,
        poisoned_external_report=None,
    )
    assert audit.severity == "warn"
    assert audit.train_eval_same_roots == []
    assert "badnet_oga" in audit.shared_attack_keys


def test_audit_cfrc_manifest_rolls_up_worst_severity(tmp_path: Path) -> None:
    ok_manifest = tmp_path / "ok_hp.json"
    ok_manifest.write_text(json.dumps(
        _manifest(replay=["D:/data/replay"], evalr=["D:/data/heldout"], datasets=["replay_only"])
    ), encoding="utf-8")
    bad_manifest = tmp_path / "bad_hp.json"
    bad_manifest.write_text(json.dumps(
        _manifest(replay=["D:/data/shared"], evalr=["D:/data/shared"], datasets=["wanet_oga"])
    ), encoding="utf-8")
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps(_report(["badnet_oga", "wanet_oga"])), encoding="utf-8")
    cfrc = {
        "entries": [
            {
                "name": "ok_arm",
                "hybrid_manifest": str(ok_manifest),
                "defended_external": str(rep),
                "poisoned_external": str(rep),
            },
            {
                "name": "bad_arm",
                "hybrid_manifest": str(bad_manifest),
                "defended_external": str(rep),
                "poisoned_external": str(rep),
            },
        ]
    }
    out = audit_cfrc_manifest(cfrc_manifest=cfrc)
    assert out["worst_severity"] == "blocked"
    severities = {row["arm"]: row["severity"] for row in out["entries"]}
    assert severities["ok_arm"] == "ok"
    assert severities["bad_arm"] == "blocked"


def test_cli_fail_on_blocked(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "bad_hp.json"
    bad_manifest.write_text(json.dumps(
        _manifest(replay=["D:/data/x"], evalr=["D:/data/x"], datasets=["badnet_oga"])
    ), encoding="utf-8")
    report = tmp_path / "rep.json"
    report.write_text(json.dumps(_report(["badnet_oga"])), encoding="utf-8")
    cfrc = tmp_path / "cfrc.json"
    cfrc.write_text(json.dumps({
        "entries": [{
            "name": "bad",
            "hybrid_manifest": str(bad_manifest),
            "defended_external": str(report),
            "poisoned_external": str(report),
        }]
    }), encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_leakage_audit.py",
            "--cfrc-manifest", str(cfrc),
            "--out", str(out),
            "--fail-on-blocked",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3, proc.stderr
    assert (out / "t0_leakage_audit.json").exists()
    assert (out / "T0_LEAKAGE_AUDIT.md").exists()
