"""Tests for Patch E: PGBD paired view must match the active detox phase.

Before Patch E, ``make_pgbd_attack_view`` only supported global perturbations
(green / sinusoidal / warp) and every phase shared the ``mixed`` combination.
The resulting paired-displacement loss pushed features against perturbations
that did not match ``attack_zoo``'s BadNet patch, so OGA/ODA phases were
training against an irrelevant view.  Patch E adds a ``badnet_patch_view`` and
an ``infer_pgbd_mode_from_phase`` helper, and wires the hybrid pipeline so each
phase automatically uses the correct view unless the caller explicitly picks
one.
"""

from __future__ import annotations

import torch

from model_security_gate.detox.pgbd_od import (
    badnet_patch_view,
    infer_pgbd_mode_from_phase,
    make_pgbd_attack_view,
)


def test_infer_mode_maps_phase_names() -> None:
    assert infer_pgbd_mode_from_phase("oga_hardening") == "badnet"
    assert infer_pgbd_mode_from_phase("oda_hardening") == "badnet_object_attached"
    assert infer_pgbd_mode_from_phase("wanet_hardening") == "warp"
    assert infer_pgbd_mode_from_phase("semantic_hardening") == "green"
    assert infer_pgbd_mode_from_phase("clean_anchor") == "mixed"
    assert infer_pgbd_mode_from_phase("clean_recovery") == "mixed"


def test_infer_mode_falls_back_on_unknown() -> None:
    assert infer_pgbd_mode_from_phase("") == "mixed"
    assert infer_pgbd_mode_from_phase(None) == "mixed"
    # token-based fallback
    assert infer_pgbd_mode_from_phase("hard_oda_phase_extra") == "badnet_object_attached"
    assert infer_pgbd_mode_from_phase("custom_oga_sprint") == "badnet"


def test_badnet_patch_view_places_in_bottom_right_by_default() -> None:
    img = torch.ones(1, 3, 32, 32) * 0.5
    out = badnet_patch_view(img, patch_frac=0.25, placement="bottom_right")
    assert out.shape == img.shape
    # Difference must be concentrated in the bottom-right quadrant.
    diff = (out - img).abs().sum(dim=1).squeeze(0)
    bottom_right = diff[16:, 16:].sum()
    top_left = diff[:16, :16].sum()
    assert bottom_right > top_left * 2


def test_badnet_patch_view_follows_object_attached_box() -> None:
    img = torch.ones(1, 3, 64, 64) * 0.5
    # Helmet bbox near the top-left quadrant.
    box = torch.tensor([[8.0, 8.0, 28.0, 28.0]])
    out = badnet_patch_view(img, patch_frac=0.2, placement="object_attached", box_xyxy=box)
    diff = (out - img).abs().sum(dim=1).squeeze(0)
    top_left = diff[:32, :32].sum()
    bottom_right = diff[32:, 32:].sum()
    assert top_left > bottom_right * 2


def test_make_view_dispatches_to_badnet() -> None:
    img = torch.ones(2, 3, 24, 24) * 0.5
    box = torch.tensor([[4.0, 4.0, 16.0, 16.0], [8.0, 8.0, 20.0, 20.0]])
    out = make_pgbd_attack_view(img, mode="badnet_object_attached", badnet_box_xyxy=box, badnet_patch_frac=0.2)
    assert out.shape == img.shape
    assert not torch.allclose(out, img)


def test_make_view_badnet_corner_mode() -> None:
    img = torch.ones(1, 3, 24, 24) * 0.5
    out = make_pgbd_attack_view(img, mode="badnet", badnet_patch_frac=0.2)
    assert out.shape == img.shape
    assert not torch.allclose(out, img)


def test_make_view_existing_modes_unchanged() -> None:
    # Use a non-uniform image so warp/blend are actually observable.
    torch.manual_seed(0)
    img = torch.rand(1, 3, 24, 24)
    # Sanity: the previously supported modes still run and change the image.
    for mode in ("green", "blend", "warp", "mixed"):
        out = make_pgbd_attack_view(img, mode=mode)
        assert out.shape == img.shape
        assert not torch.allclose(out, img, atol=1e-6), f"mode {mode} produced a no-op"


def test_hybrid_purify_picks_view_from_phase(monkeypatch) -> None:
    """_run_feature_purifier_phase should resolve pgbd_view_mode from phase_name."""
    from model_security_gate.detox import hybrid_purify_train as hpt

    # Stub run_strong_detox_training so we can inspect the config it receives.
    captured: dict = {}

    def fake_run(fcfg):
        captured["cfg"] = fcfg
        return {"best_model": str(fcfg.out_dir) + "/fake.pt", "final_model": str(fcfg.out_dir) + "/fake.pt"}

    monkeypatch.setattr(hpt, "run_strong_detox_training", fake_run)

    cfg = hpt.HybridPurifyConfig(
        pgbd_view_mode="mixed",  # default, should be auto-resolved per phase
        run_phase_finetune=False,
        run_feature_purifier=True,
    )

    # oda_hardening phase ⇒ should auto-pick badnet_object_attached.
    result = hpt._run_feature_purifier_phase(
        model="model.pt",
        teacher_model="teacher.pt",
        data_yaml="fake.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oda_hardening",
        cfg=cfg,
    )
    assert result["pgbd_view_mode"] == "badnet_object_attached"
    assert captured["cfg"].pgbd_view_mode == "badnet_object_attached"

    # oga_hardening ⇒ badnet corner
    result = hpt._run_feature_purifier_phase(
        model="model.pt",
        teacher_model="teacher.pt",
        data_yaml="fake.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oga_hardening",
        cfg=cfg,
    )
    assert result["pgbd_view_mode"] == "badnet"


def test_hybrid_purify_respects_explicit_mode(monkeypatch) -> None:
    """When caller explicitly sets a non-default mode, keep it."""
    from model_security_gate.detox import hybrid_purify_train as hpt

    def fake_run(fcfg):
        return {"best_model": str(fcfg.out_dir) + "/fake.pt", "final_model": str(fcfg.out_dir) + "/fake.pt"}

    monkeypatch.setattr(hpt, "run_strong_detox_training", fake_run)

    cfg = hpt.HybridPurifyConfig(pgbd_view_mode="green", run_phase_finetune=False)
    result = hpt._run_feature_purifier_phase(
        model="model.pt",
        teacher_model="teacher.pt",
        data_yaml="fake.yaml",
        out_dir="/tmp/out",
        target_ids=[0],
        phase_name="oda_hardening",
        cfg=cfg,
    )
    # Explicit "green" should survive even in the ODA phase.
    assert result["pgbd_view_mode"] == "green"
