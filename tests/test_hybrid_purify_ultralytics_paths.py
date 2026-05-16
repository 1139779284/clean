"""Regression tests for Ultralytics path handling in Hybrid-PURIFY-OD.

Ultralytics interprets a relative ``project`` argument under its own
``runs/detect/`` root, which used to cause ``_run_clean_recovery_finetune``
and ``_run_phase_finetune`` to save weights to a different folder than
``find_ultralytics_weight`` searched.  These tests assert both helpers now
convert ``out_project`` to an absolute path before calling Ultralytics, and
look up the resulting weights under the same absolute root.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from model_security_gate.detox import hybrid_purify_train as hpt


class _Recorder:
    """Pretend to be ``train_counterfactual_finetune`` + create weights.

    Captures the ``output_project`` it was called with so the test can check
    it is absolute, and places a ``best.pt`` / ``last.pt`` where the caller
    expects to find them.
    """

    def __init__(self, sub_name: str) -> None:
        self.sub_name = sub_name
        self.captured_project: Path | None = None

    def __call__(self, *, base_model, data_yaml, output_project, name, **kwargs) -> dict:
        self.captured_project = Path(output_project)
        weights_dir = self.captured_project / name / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"fake-weights")
        (weights_dir / "last.pt").write_bytes(b"fake-weights")
        return {"results": True}


def _cfg() -> hpt.HybridPurifyConfig:
    return hpt.HybridPurifyConfig(
        imgsz=416,
        batch=8,
        device="cpu",
        recovery_epochs=1,
        phase_epochs=1,
        feature_epochs=0,
        num_workers=0,
    )


def test_clean_recovery_uses_absolute_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder("clean_recovery")
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    relative_out = Path("runs/fake_project_relative")
    result = hpt._run_clean_recovery_finetune(
        model=tmp_path / "fake_base.pt",
        data_yaml=tmp_path / "fake.yaml",
        out_project=relative_out,
        cfg=_cfg(),
        epochs=1,
    )
    # The recorder must have received an absolute path, not the relative input.
    assert recorder.captured_project is not None
    assert recorder.captured_project.is_absolute()
    assert result.is_absolute()
    assert result.name == "best.pt"
    # The returned weight must live under the same absolute project.
    assert result.parent.parent.parent == recorder.captured_project


def test_phase_finetune_uses_absolute_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder("phase_finetune")
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    relative_out = Path("runs/another_fake_relative")
    candidates = hpt._run_phase_finetune(
        model=tmp_path / "fake_base.pt",
        data_yaml=tmp_path / "fake.yaml",
        out_project=relative_out,
        cfg=_cfg(),
        phase_name="oga_hardening",
        epochs=1,
    )
    assert recorder.captured_project is not None
    assert recorder.captured_project.is_absolute()
    assert len(candidates) >= 1
    for path in candidates:
        assert path.is_absolute()
        assert recorder.captured_project in path.parents
