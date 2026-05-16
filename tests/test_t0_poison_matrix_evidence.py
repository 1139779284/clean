from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from model_security_gate.t0.poison_matrix_evidence import (
    PoisonMatrixEvidenceConfig,
    build_poison_matrix_evidence,
    write_poison_matrix_evidence,
)


def _write_report(path: Path, attack: str, asr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "max_asr": asr,
                    "mean_asr": asr,
                    "asr_matrix": {f"suite::{attack}": asr},
                    "top_attacks": [{"attack": attack, "asr": asr, "n": 100}],
                }
            }
        ),
        encoding="utf-8",
    )


def test_poison_matrix_evidence_accepts_strong_core(tmp_path: Path):
    entries = []
    for attack, asr in [("badnet_oga_corner", 0.9), ("semantic_cleanlabel", 0.8), ("wanet_oga", 0.3)]:
        report = tmp_path / "eval" / attack / "external_hard_suite_asr.json"
        weight = tmp_path / "training" / attack / "weights" / "best.pt"
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"fake")
        _write_report(report, attack, asr)
        entries.append({"attack": attack, "run": attack, "report": str(report), "weights": str(weight)})
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        root=".",
        cfg=PoisonMatrixEvidenceConfig(expected_attacks=("badnet_oga_corner", "semantic_cleanlabel", "wanet_oga")),
    )
    assert evidence["status"] == "passed"


def test_poison_matrix_evidence_blocks_weak_attack(tmp_path: Path):
    report = tmp_path / "eval" / "wanet_oga" / "external_hard_suite_asr.json"
    weight = tmp_path / "training" / "wanet_oga" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"fake")
    _write_report(report, "wanet_oga", 0.04)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"entries": [{"attack": "wanet_oga", "run": "wanet_oga", "report": str(report), "weights": str(weight)}]}),
        encoding="utf-8",
    )

    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        cfg=PoisonMatrixEvidenceConfig(expected_attacks=("wanet_oga",), min_usable_asr=0.05),
    )
    assert evidence["status"] == "blocked"
    assert evidence["coverage"]["wanet_oga"]["blocked"]


def test_poison_matrix_evidence_writes_reports(tmp_path: Path):
    report = tmp_path / "eval" / "badnet_oga_corner" / "external_hard_suite_asr.json"
    weight = tmp_path / "training" / "badnet_oga_corner" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"fake")
    _write_report(report, "badnet_oga_corner", 0.7)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"entries": [{"attack": "badnet_oga_corner", "run": "badnet", "report": str(report), "weights": str(weight)}]}),
        encoding="utf-8",
    )
    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        cfg=PoisonMatrixEvidenceConfig(expected_attacks=("badnet_oga_corner",)),
    )
    json_path, md_path = write_poison_matrix_evidence(tmp_path / "out", evidence)
    assert json_path.exists()
    assert md_path.exists()
    assert "badnet_oga_corner" in md_path.read_text(encoding="utf-8")


def test_poison_matrix_full_factorial_blocks_missing_cells(tmp_path: Path):
    report = tmp_path / "eval" / "badnet_oga_corner_pr0100_seed1" / "external_hard_suite_asr.json"
    weight = tmp_path / "training" / "badnet_oga_corner_pr0100_seed1" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"fake")
    _write_report(report, "badnet_oga_corner", 0.8)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "attack": "badnet_oga_corner",
                        "run": "badnet_oga_corner_pr0100_seed1",
                        "poison_rate": 0.01,
                        "seed": 1,
                        "report": str(report),
                        "weights": str(weight),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        cfg=PoisonMatrixEvidenceConfig(
            expected_attacks=("badnet_oga_corner",),
            expected_seeds=(1, 2),
            expected_poison_rates=(0.01, 0.03),
            require_full_factorial=True,
        ),
    )
    assert evidence["status"] == "blocked"
    assert len(evidence["coverage"]["badnet_oga_corner"]["missing_cells"]) == 3
    assert "missing full-factorial" in evidence["blocked_reasons"][0]


def test_poison_matrix_full_factorial_passes_complete_cells(tmp_path: Path):
    entries = []
    for seed in [1, 2]:
        for poison_rate in [0.01, 0.03]:
            run = f"badnet_oga_corner_pr{int(poison_rate * 10000):04d}_seed{seed}"
            report = tmp_path / "eval" / run / "external_hard_suite_asr.json"
            weight = tmp_path / "training" / run / "weights" / "best.pt"
            weight.parent.mkdir(parents=True, exist_ok=True)
            weight.write_bytes(b"fake")
            _write_report(report, "badnet_oga_corner", 0.6)
            entries.append(
                {
                    "attack": "badnet_oga_corner",
                    "run": run,
                    "poison_rate": poison_rate,
                    "seed": seed,
                    "report": str(report),
                    "weights": str(weight),
                }
            )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        cfg=PoisonMatrixEvidenceConfig(
            expected_attacks=("badnet_oga_corner",),
            expected_seeds=(1, 2),
            expected_poison_rates=(0.01, 0.03),
            require_full_factorial=True,
        ),
    )
    assert evidence["status"] == "passed"
    assert evidence["coverage"]["badnet_oga_corner"]["missing_cells"] == []


def test_poison_matrix_full_factorial_present_accepts_low_asr_cell(tmp_path: Path):
    report = tmp_path / "eval" / "badnet_oga_corner_pr0100_seed1" / "external_hard_suite_asr.json"
    weight = tmp_path / "training" / "badnet_oga_corner_pr0100_seed1" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"fake")
    _write_report(report, "badnet_oga_corner", 0.01)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "attack": "badnet_oga_corner",
                        "run": "badnet_oga_corner_pr0100_seed1",
                        "poison_rate": 0.01,
                        "seed": 1,
                        "report": str(report),
                        "weights": str(weight),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence = build_poison_matrix_evidence(
        summary_json=summary,
        cfg=PoisonMatrixEvidenceConfig(
            expected_attacks=("badnet_oga_corner",),
            expected_seeds=(1,),
            expected_poison_rates=(0.01,),
            require_full_factorial=True,
            full_factorial_cell_acceptance="present",
            require_any_strong=False,
        ),
    )
    assert evidence["status"] == "passed"
    assert evidence["entries"][0]["status"] == "blocked"


def test_poison_matrix_completion_plan_cli(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"entries": []}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "expected_attacks:",
                "  - badnet_oga_corner",
                "expected_seeds:",
                "  - 1",
                "  - 2",
                "expected_poison_rates:",
                "  - 0.01",
                "min_primary_asr: 0.2",
                "min_usable_asr: 0.05",
                "require_full_factorial: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/t0_poison_matrix_completion_plan.py",
            "--summary-json",
            str(summary),
            "--evidence-config",
            str(config),
            "--out",
            str(out),
            "--train-out",
            str(tmp_path / "train"),
            "--clean-root",
            str(tmp_path / "clean"),
            "--base-model",
            "yolo26n.pt",
            "--data-yaml",
            str(tmp_path / "data.yaml"),
            "--eval-roots",
            str(tmp_path / "eval_suite"),
        ],
        check=True,
    )
    plan = json.loads((out / "t0_poison_matrix_completion_plan.json").read_text(encoding="utf-8"))
    assert plan["n_planned"] == 2
    assert "train_t0_poison_models_yolo.py" in plan["cells"][0]["train_command"]
    assert (out / "t0_poison_matrix_merged_summary_planned.json").exists()
