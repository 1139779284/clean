"""Tests for Patch B: external replay floor in ``_build_phase_dataset``.

When ``external_failure_replay=True`` the replay is filtered to previously
observed failure rows.  As detox progresses, the failure list shrinks
because the defended model recovers some attacks.  Without a floor, the
phase dataset then contains only a handful of real attack samples while
hundreds of synthetic samples dominate training.  Patch B adds
``external_replay_floor_per_attack`` so each phase always replays at
least N real samples per attack on top of the failure-only pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from model_security_gate.detox import asr_closed_loop_train as mod
from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig
from model_security_gate.detox.asr_closed_loop_train import (
    ASRClosedLoopConfig,
    ClosedLoopPhase,
    _build_phase_dataset,
)


class _FakeDataset:
    """Minimal stand-in for ExternalAttackDataset (only presence matters)."""

    name = "badnet_oda"


def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: List[Dict[str, Any]]) -> None:
    """Stub out heavy dependencies so _build_phase_dataset stays pure-logic."""

    def fake_build_asr_aware_yolo_dataset(**kwargs: Any) -> Path:
        phase_dir = Path(kwargs["output_dir"])
        phase_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = phase_dir / "data.yaml"
        yaml_path.write_text("names: [helmet]\n")
        return yaml_path

    def fake_append(**kwargs: Any) -> Dict[str, Any]:
        calls.append(dict(kwargs))
        if bool(kwargs.get("failure_only")):
            # Simulate a nearly-empty failure list: 2 real samples.
            return {"added": 2, "skipped": 0, "by_attack": {"badnet_oda": 2}}
        # Baseline top-up adds ``max_images_per_attack`` samples per attack.
        cap = int(kwargs.get("max_images_per_attack") or 0)
        return {"added": cap, "skipped": 0, "by_attack": {"badnet_oda": cap}}

    def fake_write_json(path: Path, payload: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{}")

    monkeypatch.setattr(mod, "build_asr_aware_yolo_dataset", fake_build_asr_aware_yolo_dataset)
    monkeypatch.setattr(mod, "append_external_replay_samples", fake_append)
    monkeypatch.setattr(mod, "write_json", fake_write_json)


def _phase() -> ClosedLoopPhase:
    spec = AttackTransformConfig(name="badnet_oda", kind="badnet", goal="oda")
    return ClosedLoopPhase(name="oda_hardening", attacks=(spec,), epochs=1, attack_repeat=1, clean_repeat=2, replay_external=True)


def test_replay_floor_adds_samples_when_failures_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fakes(monkeypatch, calls)

    cfg = ASRClosedLoopConfig(
        external_failure_replay=True,
        external_failure_replay_repeat=4,
        external_replay_floor_per_attack=20,
        external_replay_max_images_per_attack=250,
    )

    _build_phase_dataset(
        phase=_phase(),
        cycle=2,
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        names={0: "helmet"},
        target_ids=[0],
        cfg=cfg,
        replay_datasets=[_FakeDataset()],
        failure_rows=[],  # nothing to replay from failures
    )

    # Two calls to append_external_replay_samples: one failure_only, one top-up.
    assert len(calls) == 2, f"expected 2 calls, got {len(calls)}"
    assert calls[0]["failure_only"] is True
    assert calls[1]["failure_only"] is False
    assert int(calls[1]["max_images_per_attack"]) == 20
    assert int(calls[1]["repeat"]) == 1
    assert calls[1]["failure_rows"] is None

    # Manifest is written once; check that merged stats reflect the floor.
    manifest = tmp_path / "01_cycle_02_oda_hardening_dataset" / "phase_manifest.json"
    assert manifest.exists()


def test_replay_floor_zero_preserves_legacy_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fakes(monkeypatch, calls)

    cfg = ASRClosedLoopConfig(
        external_failure_replay=True,
        external_failure_replay_repeat=4,
        external_replay_floor_per_attack=0,  # legacy default
        external_replay_max_images_per_attack=250,
    )

    _build_phase_dataset(
        phase=_phase(),
        cycle=2,
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        names={0: "helmet"},
        target_ids=[0],
        cfg=cfg,
        replay_datasets=[_FakeDataset()],
        failure_rows=[],
    )

    assert len(calls) == 1
    assert calls[0]["failure_only"] is True


def test_replay_floor_repeat_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    _install_fakes(monkeypatch, calls)

    cfg = ASRClosedLoopConfig(
        external_failure_replay=True,
        external_failure_replay_repeat=4,
        external_replay_floor_per_attack=20,
        external_replay_floor_repeat=5,
    )

    _build_phase_dataset(
        phase=_phase(),
        cycle=2,
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        names={0: "helmet"},
        target_ids=[0],
        cfg=cfg,
        replay_datasets=[_FakeDataset()],
        failure_rows=[],
    )

    assert len(calls) == 2
    assert calls[1]["failure_only"] is False
    assert int(calls[1]["max_images_per_attack"]) == 20
    assert int(calls[1]["repeat"]) == 5


def test_replay_floor_merges_by_attack_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """added and by_attack should be the sum of both passes."""

    import json

    calls: List[Dict[str, Any]] = []

    def fake_build_asr_aware_yolo_dataset(**kwargs: Any) -> Path:
        phase_dir = Path(kwargs["output_dir"])
        phase_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = phase_dir / "data.yaml"
        yaml_path.write_text("names: [helmet]\n")
        return yaml_path

    def fake_append(**kwargs: Any) -> Dict[str, Any]:
        calls.append(dict(kwargs))
        if bool(kwargs.get("failure_only")):
            return {"added": 6, "skipped": 1, "by_attack": {"badnet_oda": 6}}
        cap = int(kwargs.get("max_images_per_attack") or 0)
        return {"added": cap, "skipped": 3, "by_attack": {"badnet_oda": cap}}

    captured_manifest: Dict[str, Any] = {}

    def fake_write_json(path: Path, payload: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        captured_manifest.update(payload)
        Path(path).write_text(json.dumps(payload, default=str))

    monkeypatch.setattr(mod, "build_asr_aware_yolo_dataset", fake_build_asr_aware_yolo_dataset)
    monkeypatch.setattr(mod, "append_external_replay_samples", fake_append)
    monkeypatch.setattr(mod, "write_json", fake_write_json)

    cfg = ASRClosedLoopConfig(
        external_failure_replay=True,
        external_failure_replay_repeat=4,
        external_replay_floor_per_attack=20,
    )

    _build_phase_dataset(
        phase=_phase(),
        cycle=2,
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        names={0: "helmet"},
        target_ids=[0],
        cfg=cfg,
        replay_datasets=[_FakeDataset()],
        failure_rows=[],
    )

    stats = captured_manifest.get("replay_stats", {})
    assert int(stats.get("added", 0)) >= 20  # floor guarantee
    assert int(stats.get("added", 0)) == 6 + 20
    assert int(stats.get("skipped", 0)) == 1 + 3
    assert stats["by_attack"]["badnet_oda"] == 6 + 20
    assert int(stats["external_replay_floor_per_attack"]) == 20
