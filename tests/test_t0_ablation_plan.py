"""Tests for the T0 detox ablation planner.

These tests stay off the GPU; they only exercise command generation,
manifest parsing, CFRC manifest emission, and CLI wiring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_security_gate.t0.ablation_plan import (
    AblationArm,
    DetoxAblationSpec,
    build_arm_train_command,
    build_cfrc_manifest,
    default_contribution_1_arms,
    hybrid_manifest_to_defense_entry,
    plan_runs,
    render_runbook_markdown,
    write_ablation_plan,
)


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------


def _spec(tmp_path: Path) -> DetoxAblationSpec:
    return DetoxAblationSpec(
        poisoned_model_id="best_2_poisoned",
        poisoned_model_path="D:/models/best_2.pt",
        teacher_model="D:/models/teacher.pt",
        data_yaml="D:/data/data.yaml",
        images="D:/data/images/train",
        labels="D:/data/labels/train",
        target_classes=("helmet",),
        external_eval_roots=("D:/bench/suite_corrected",),
        external_replay_roots=("D:/bench/suite_large",),
        cycles=2,
        phase_epochs=1,
        feature_epochs=1,
        recovery_epochs=1,
        device="0",
        out_root=str(tmp_path / "runs"),
    )


def test_build_arm_train_command_static_arm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm = AblationArm(name="static_lambda", use_lagrangian_controller=False)
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert cmd[0] == "python"
    assert "scripts/hybrid_purify_detox_yolo.py" in cmd
    assert "--model" in cmd and "D:/models/best_2.pt" in cmd
    assert "--data-yaml" in cmd and "D:/data/data.yaml" in cmd
    assert "--teacher-model" in cmd and "D:/models/teacher.pt" in cmd
    assert "--target-classes" in cmd and "helmet" in cmd
    assert "--external-eval-roots" in cmd and "D:/bench/suite_corrected" in cmd
    assert "--external-replay-roots" in cmd and "D:/bench/suite_large" in cmd
    assert "--use-lagrangian-controller" not in cmd


def test_build_arm_train_command_lagrangian_arm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm = AblationArm(name="lagrangian_lambda", use_lagrangian_controller=True)
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert "--use-lagrangian-controller" in cmd


def test_build_arm_train_command_passes_hybrid_config(tmp_path: Path) -> None:
    spec = DetoxAblationSpec(
        **{**_spec(tmp_path).__dict__, "hybrid_config": "configs/mask_bd_v2_hybrid_purify.yaml"}
    )
    arm = AblationArm(name="static_lambda", use_lagrangian_controller=False)
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert cmd[2:4] == ["--config", "configs/mask_bd_v2_hybrid_purify.yaml"]


def test_build_arm_train_command_passes_extra_cli(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm = AblationArm(
        name="lagrangian_aggressive",
        use_lagrangian_controller=True,
        extra_cli=("--aggressive-mode",),
    )
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert "--aggressive-mode" in cmd


def test_build_arm_train_command_omits_teacher_when_none(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec_no_teacher = DetoxAblationSpec(**{**spec.__dict__, "teacher_model": None})
    arm = AblationArm(name="no_teacher", use_lagrangian_controller=False)
    cmd = build_arm_train_command(spec=spec_no_teacher, arm=arm)
    assert "--teacher-model" not in cmd


def test_build_arm_train_command_emits_smoke_sizing(tmp_path: Path) -> None:
    spec = DetoxAblationSpec(
        **{
            **_spec(tmp_path).__dict__,
            "max_images": 800,
            "eval_max_images": 100,
            "external_eval_max_images_per_attack": 40,
            "external_replay_max_images_per_attack": 80,
            "selection_max_map_drop": 0.08,
            "no_pre_prune": True,
        }
    )
    arm = AblationArm(name="static_lambda", use_lagrangian_controller=False)
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert "--max-images" in cmd and "800" in cmd
    assert "--eval-max-images" in cmd and "100" in cmd
    assert "--external-eval-max-images-per-attack" in cmd and "40" in cmd
    assert "--external-replay-max-images-per-attack" in cmd and "80" in cmd
    assert "--selection-max-map-drop" in cmd and "0.08" in cmd
    assert "--no-pre-prune" in cmd


def test_build_arm_train_command_emits_config_overrides(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm = AblationArm(
        name="no_oda_matched",
        use_lagrangian_controller=True,
        config_overrides={
            "aggressive_lambda_oda_matched": 0.0,
            "aggressive_mode": True,
            "aggressive_mode_disabled_example": False,
        },
    )
    cmd = build_arm_train_command(spec=spec, arm=arm)
    assert "--aggressive-lambda-oda-matched" in cmd
    assert "0.0" in cmd
    assert "--aggressive-mode" in cmd
    # Bool False should be omitted.
    assert "--aggressive-mode-disabled-example" not in cmd
    assert "--use-lagrangian-controller" in cmd


def test_build_arm_train_command_emits_list_config_override(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm = AblationArm(
        name="multi",
        use_lagrangian_controller=False,
        config_overrides={"target_classes": ["helmet", "head"]},
    )
    cmd = build_arm_train_command(spec=spec, arm=arm)
    idx = cmd.index("--target-classes")
    # Arm-level target-classes appears AFTER the spec-level target-classes
    # because arm overrides are appended.  Ensure all values are present.
    assert "helmet" in cmd
    assert "head" in cmd


def test_build_arm_train_command_omits_smoke_flags_when_absent(tmp_path: Path) -> None:
    spec = _spec(tmp_path)  # None defaults; no extra flags should appear.
    arm = AblationArm(name="full", use_lagrangian_controller=True)
    cmd = build_arm_train_command(spec=spec, arm=arm)
    for flag in (
        "--max-images",
        "--eval-max-images",
        "--external-eval-max-images-per-attack",
        "--external-replay-max-images-per-attack",
        "--selection-max-map-drop",
        "--no-pre-prune",
    ):
        assert flag not in cmd


# ---------------------------------------------------------------------------
# Plan runs + on-disk existence
# ---------------------------------------------------------------------------


def test_plan_runs_reports_missing_artifacts(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runs = plan_runs(spec, list(default_contribution_1_arms()))
    assert len(runs) == 2
    for run in runs:
        assert not run.exists["hybrid_manifest"]
        assert not run.exists["poisoned_external"]
        assert not run.exists["defended_external"]


def test_plan_runs_uses_manifest_paths_when_present(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm_dir = Path(spec.out_root) / "static_lambda"
    arm_dir.mkdir(parents=True, exist_ok=True)
    poisoned = arm_dir / "eval_00_before_external" / "external_hard_suite_asr.json"
    defended = arm_dir / "eval_cycle_01_external" / "external_hard_suite_asr.json"
    poisoned.parent.mkdir(parents=True, exist_ok=True)
    defended.parent.mkdir(parents=True, exist_ok=True)
    poisoned.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.5}}}), encoding="utf-8")
    defended.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.05}}}), encoding="utf-8")
    manifest = arm_dir / "hybrid_purify_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "before_eval": {
                    "external_json": str(poisoned),
                    "clean_metrics": {"map50_95": 0.30},
                },
                "best": {
                    "external_json": str(defended),
                    "clean_metrics": {"map50_95": 0.29},
                },
            }
        ),
        encoding="utf-8",
    )
    runs = plan_runs(spec, list(default_contribution_1_arms()))
    static_run = next(r for r in runs if r.arm == "static_lambda")
    assert static_run.exists["hybrid_manifest"] is True
    assert static_run.exists["poisoned_external"] is True
    assert static_run.exists["defended_external"] is True
    assert Path(static_run.clean_before).exists()
    assert Path(static_run.clean_after).exists()


# ---------------------------------------------------------------------------
# Hybrid manifest -> CFRC entry conversion
# ---------------------------------------------------------------------------


def test_hybrid_manifest_to_defense_entry_skips_missing_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "hybrid_purify_manifest.json"
    manifest.write_text(json.dumps({"before_eval": {}, "best": {}}), encoding="utf-8")
    out = hybrid_manifest_to_defense_entry(
        arm_name="static_lambda",
        poisoned_model_id="x",
        hybrid_manifest_path=manifest,
    )
    assert out is None


def test_hybrid_manifest_to_defense_entry_writes_clean_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "hybrid_purify_manifest.json"
    poisoned = tmp_path / "poisoned.json"
    defended = tmp_path / "defended.json"
    poisoned.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.5}}}), encoding="utf-8")
    defended.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.05}}}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "before_eval": {
                    "external_json": str(poisoned),
                    "clean_metrics": {"map50_95": 0.30},
                },
                "best": {
                    "external_json": str(defended),
                    "clean_metrics": {"map50_95": 0.29},
                },
            }
        ),
        encoding="utf-8",
    )
    entry = hybrid_manifest_to_defense_entry(
        arm_name="lagrangian_lambda",
        poisoned_model_id="best_2_poisoned",
        hybrid_manifest_path=manifest,
    )
    assert entry is not None
    assert entry["name"] == "lagrangian_lambda"
    assert entry["defense"] == "lagrangian_lambda"
    assert entry["poisoned_model_id"] == "best_2_poisoned"
    assert entry["poisoned_external"] == str(poisoned)
    assert entry["defended_external"] == str(defended)
    assert Path(entry["clean_before"]).exists()
    assert Path(entry["clean_after"]).exists()


def test_hybrid_manifest_to_defense_entry_overwrites_stale_clean_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "hybrid_purify_manifest.json"
    poisoned = tmp_path / "poisoned.json"
    defended = tmp_path / "defended.json"
    poisoned.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.5}}}), encoding="utf-8")
    defended.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.05}}}), encoding="utf-8")
    (tmp_path / "clean_before.json").write_text(json.dumps({"map50_95": 0.99}), encoding="utf-8")
    (tmp_path / "clean_after.json").write_text(json.dumps({"map50_95": 0.01}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "before_eval": {
                    "external_json": str(poisoned),
                    "clean_metrics": {"map50_95": 0.30},
                },
                "best": {
                    "external_json": str(defended),
                    "clean_metrics": {"map50_95": 0.27},
                },
            }
        ),
        encoding="utf-8",
    )
    entry = hybrid_manifest_to_defense_entry(
        arm_name="static_lambda",
        poisoned_model_id="best_2_poisoned",
        hybrid_manifest_path=manifest,
    )
    assert entry is not None
    clean_before = json.loads((tmp_path / "clean_before.json").read_text(encoding="utf-8"))
    clean_after = json.loads((tmp_path / "clean_after.json").read_text(encoding="utf-8"))
    assert clean_before["map50_95"] == pytest.approx(0.30)
    assert clean_after["map50_95"] == pytest.approx(0.27)


# ---------------------------------------------------------------------------
# CFRC manifest: entries vs skipped
# ---------------------------------------------------------------------------


def test_build_cfrc_manifest_splits_ready_and_pending(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arm_dir = Path(spec.out_root) / "static_lambda"
    arm_dir.mkdir(parents=True, exist_ok=True)
    poisoned = arm_dir / "poisoned.json"
    defended = arm_dir / "defended.json"
    poisoned.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.5}}}), encoding="utf-8")
    defended.write_text(json.dumps({"summary": {"asr_matrix": {"badnet_oga": 0.05}}}), encoding="utf-8")
    (arm_dir / "hybrid_purify_manifest.json").write_text(
        json.dumps(
            {
                "before_eval": {"external_json": str(poisoned), "clean_metrics": {"map50_95": 0.30}},
                "best": {"external_json": str(defended), "clean_metrics": {"map50_95": 0.29}},
            }
        ),
        encoding="utf-8",
    )
    runs = plan_runs(spec, list(default_contribution_1_arms()))
    cfrc = build_cfrc_manifest(poisoned_model_id=spec.poisoned_model_id, runs=runs)
    assert len(cfrc["entries"]) == 1
    assert cfrc["entries"][0]["name"] == "static_lambda"
    assert len(cfrc["skipped"]) == 1
    assert cfrc["skipped"][0]["arm"] == "lagrangian_lambda"


def test_write_ablation_plan_emits_json_and_runbook(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arms = list(default_contribution_1_arms())
    runs = plan_runs(spec, arms)
    cfrc = build_cfrc_manifest(poisoned_model_id=spec.poisoned_model_id, runs=runs)
    out_dir = tmp_path / "plan_out"
    written = write_ablation_plan(out_dir, spec=spec, arms=arms, runs=runs, cfrc_manifest=cfrc)
    assert written["plan"].exists()
    assert written["runbook"].exists()
    assert written["cfrc_manifest"].exists()
    md = written["runbook"].read_text(encoding="utf-8")
    assert "static_lambda" in md
    assert "lagrangian_lambda" in md
    assert "scripts/hybrid_purify_detox_yolo.py" in md


def test_render_markdown_contains_exists_map(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    arms = list(default_contribution_1_arms())
    runs = plan_runs(spec, arms)
    md = render_runbook_markdown(spec=spec, arms=arms, runs=runs)
    assert "MISSING" in md  # artifacts not yet produced


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generates_plan_runbook_and_cfrc_manifest(tmp_path: Path) -> None:
    spec_yaml = tmp_path / "spec.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "spec:",
                "  poisoned_model_id: demo",
                "  poisoned_model_path: D:/p/best.pt",
                "  teacher_model: D:/p/teacher.pt",
                "  data_yaml: D:/d/data.yaml",
                "  images: D:/d/images",
                "  labels: D:/d/labels",
                "  target_classes: [helmet]",
                "  external_eval_roots: [D:/bench/suite]",
                "  out_root: " + str(tmp_path / "runs"),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "plan_out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_detox_ablation_plan.py",
            "--spec",
            str(spec_yaml),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads((out / "t0_detox_ablation_plan.json").read_text(encoding="utf-8"))
    assert len(plan["arms"]) == 2
    cfrc = json.loads((out / "cfrc_manifest.json").read_text(encoding="utf-8"))
    assert cfrc["entries"] == []
    assert len(cfrc["skipped"]) == 2


def test_cli_fail_on_missing(tmp_path: Path) -> None:
    spec_yaml = tmp_path / "spec.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "spec:",
                "  poisoned_model_id: demo",
                "  poisoned_model_path: D:/p/best.pt",
                "  data_yaml: D:/d/data.yaml",
                "  images: D:/d/images",
                "  labels: D:/d/labels",
                "  target_classes: [helmet]",
                "  external_eval_roots: [D:/bench/suite]",
                "  out_root: " + str(tmp_path / "runs"),
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/t0_detox_ablation_plan.py",
            "--spec",
            str(spec_yaml),
            "--out",
            str(tmp_path / "out"),
            "--fail-on-missing",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
