"""Phase finetune honours Lagrangian lambdas when provided.

After the smoke-scale ablation showed that static_lambda and
lagrangian_lambda produced identical per-attack ASR because the selected
best checkpoint came from `_run_phase_finetune` (an Ultralytics supervised
fine-tune that previously did not read the controller), we wire the
controller's current lambdas into the phase finetune's learning rate via a
per-bucket sqrt-scaled multiplier.  These tests assert that behaviour
without requiring GPU: we monkey-patch `train_counterfactual_finetune` to
capture the emitted `lr0` and confirm the scaling works as designed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_security_gate.detox import hybrid_purify_train as hpt


class _LrRecorder:
    """Fake ``train_counterfactual_finetune`` that records ``lr0``.

    Also emits a fake ``best.pt`` so ``find_ultralytics_weight`` returns
    something real.
    """

    def __init__(self) -> None:
        self.captured_lr0: float | None = None

    def __call__(self, *, base_model, data_yaml, output_project, name, **kwargs) -> dict:
        self.captured_lr0 = float(kwargs.get("lr0", 0.0))
        weights_dir = Path(output_project) / name / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"fake-weights")
        (weights_dir / "last.pt").write_bytes(b"fake-weights")
        return {}


def _cfg(**overrides) -> hpt.HybridPurifyConfig:
    defaults = dict(
        imgsz=416,
        batch=8,
        device="cpu",
        phase_epochs=1,
        feature_epochs=0,
        recovery_epochs=1,
        num_workers=0,
        lr=2e-5,
        aggressive_mode=False,
        lagrangian_lambda_max=6.0,
        lagrangian_base_scale=1.0,
        lagrangian_max_scale=4.0,
        lagrangian_min_scale=0.5,
    )
    defaults.update(overrides)
    return hpt.HybridPurifyConfig(**defaults)


def test_phase_finetune_baseline_lr_without_lagrangian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LrRecorder()
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    cfg = _cfg()
    hpt._run_phase_finetune(
        model=tmp_path / "m.pt",
        data_yaml=tmp_path / "d.yaml",
        out_project=tmp_path / "out",
        cfg=cfg,
        phase_name="oga_hardening",
    )
    # Baseline lr = cfg.lr (not aggressive, no lagrangian).
    assert recorder.captured_lr0 == pytest.approx(cfg.lr)


def test_phase_finetune_scales_lr_up_on_violated_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LrRecorder()
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    cfg = _cfg()
    # OGA bucket takes its lambda mean from badnet_oga + blend_oga (both full).
    # Per-bucket scale = 1 + (6/6) * (4 - 1) = 4.0. sqrt(4) = 2.0.
    hpt._run_phase_finetune(
        model=tmp_path / "m.pt",
        data_yaml=tmp_path / "d.yaml",
        out_project=tmp_path / "out",
        cfg=cfg,
        phase_name="oga_hardening",
        lagrangian_lambdas={"badnet_oga": 6.0, "blend_oga": 6.0},
    )
    # lr should roughly double (sqrt(4) = 2).
    assert recorder.captured_lr0 == pytest.approx(cfg.lr * 2.0)


def test_phase_finetune_passes_through_when_no_matching_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LrRecorder()
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    cfg = _cfg()
    # Phase is OGA but the observed lambdas are all for ODA; scale should
    # not fire because the OGA bucket has no entries.
    hpt._run_phase_finetune(
        model=tmp_path / "m.pt",
        data_yaml=tmp_path / "d.yaml",
        out_project=tmp_path / "out",
        cfg=cfg,
        phase_name="oga_hardening",
        lagrangian_lambdas={"badnet_oda": 5.0, "wanet_oda": 5.0},
    )
    assert recorder.captured_lr0 == pytest.approx(cfg.lr)


def test_phase_finetune_scale_clamped_at_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LrRecorder()
    monkeypatch.setattr(hpt, "train_counterfactual_finetune", recorder)
    # Configure a base_scale BELOW min_scale so the clamp has to kick in.
    cfg = _cfg(
        lagrangian_base_scale=0.2,
        lagrangian_max_scale=4.0,
        lagrangian_min_scale=0.5,
    )
    hpt._run_phase_finetune(
        model=tmp_path / "m.pt",
        data_yaml=tmp_path / "d.yaml",
        out_project=tmp_path / "out",
        cfg=cfg,
        phase_name="oga_hardening",
        lagrangian_lambdas={"badnet_oga": 0.0},
    )
    # Scale clamped to 0.5, sqrt(0.5) ~= 0.7071.
    assert recorder.captured_lr0 == pytest.approx(cfg.lr * (0.5 ** 0.5))
