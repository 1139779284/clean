"""Tests for Patch A: asr_aware_dataset triggers must align with attack_zoo.

Before Patch A, ``asr_aware_dataset.apply_attack_transform`` implemented its own
trigger ops that did not match ``attack_zoo.image_ops.apply_attack_image``.
The badnet_oda training samples placed the patch in the bottom-right corner,
while the real evaluation suite placed it on the helmet (``object_attached``).
This caused a train/eval distribution mismatch so ODA-phase fine-tuning never
taught the defended model to recover helmet confidence when the patch covers
the target.  Patch A rewires the detox-side transforms to dispatch through
``attack_zoo`` so both paths share trigger parameters by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_security_gate.detox import asr_aware_dataset as mod
from model_security_gate.detox.asr_aware_dataset import (
    AttackTransformConfig,
    apply_attack_transform,
    default_attack_suite,
)


def _solid_image(size: int = 64, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_attack_zoo_is_the_default_dispatch() -> None:
    """apply_attack_transform should route to attack_zoo when available.

    In the normal installed environment we expect _ATTACK_ZOO_AVAILABLE=True.
    If it becomes False in CI for any reason, that is a red flag by itself.
    """
    assert mod._ATTACK_ZOO_AVAILABLE, (
        "attack_zoo integration is required for detox training parity. "
        "If this fails, check that model_security_gate.attack_zoo is installed."
    )


def test_default_suite_badnet_oda_uses_object_attached() -> None:
    """Patch A default: ODA synthetic trigger must stick to the helmet box."""
    suite = {spec.name: spec for spec in default_attack_suite()}
    oda = suite["badnet_oda"]
    # ODA goal ⇒ must preserve labels (poison_positive=True) and default to
    # object_attached placement (aligns with attack_zoo eval).
    assert oda.goal == "oda"
    assert oda.poison_positive is True
    assert oda.poison_negative is False
    assert str(oda.params.get("position", "")).lower() == "object_attached"
    # Patch size should match the attack_zoo default range (0.05-0.07), not the
    # legacy 0.09 that used the wrong corner placement.
    assert 0.04 <= float(oda.params.get("patch_frac", 0.0)) <= 0.08


def test_default_suite_badnet_oga_uses_corner_placement() -> None:
    suite = {spec.name: spec for spec in default_attack_suite()}
    oga = suite["badnet_oga"]
    assert oga.goal == "oga"
    assert oga.poison_negative is True
    assert oga.poison_positive is False
    pos = str(oga.params.get("position", "")).lower()
    assert pos in {"bottom_right", "br"}
    assert 0.04 <= float(oga.params.get("patch_frac", 0.0)) <= 0.08


def test_badnet_oda_patch_lands_on_helmet_bbox() -> None:
    """Critical correctness check.

    For goal=oda with object_attached placement, the patch pixels should fall
    inside or adjacent to the supplied bbox, not in the opposite corner.
    """
    img = _solid_image(128, 128)
    spec = AttackTransformConfig(
        name="badnet_oda",
        kind="badnet_patch",
        goal="oda",
        poison_positive=True,
        poison_negative=False,
        params={"patch_frac": 0.07, "position": "object_attached"},
    )
    # Helmet bbox at image center (40, 40) to (80, 80) in a 128x128 image.
    box = (40.0, 40.0, 80.0, 80.0)
    out = apply_attack_transform(img, spec, seed=7, box_xyxy=box)
    diff = np.any(out != img, axis=-1)
    assert diff.any(), "trigger had no effect on the image"
    # Patch centroid must be near the helmet box, not in the bottom-right.
    ys, xs = np.where(diff)
    cy, cx = float(ys.mean()), float(xs.mean())
    box_cx, box_cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    dx = abs(cx - box_cx)
    dy = abs(cy - box_cy)
    # Allow some offset because object_attached places the patch near the top of
    # the helmet, not centered. Still must be clearly closer to the helmet than
    # to the bottom-right corner (which is the legacy wrong placement).
    br_dx = abs(cx - 120.0)
    br_dy = abs(cy - 120.0)
    assert (dx + dy) < (br_dx + br_dy), (
        f"patch landed near bottom-right ({cx:.1f},{cy:.1f}) "
        f"rather than on the helmet bbox center ({box_cx:.1f},{box_cy:.1f})"
    )


def test_badnet_oga_patch_lands_in_corner() -> None:
    """OGA goal should still place the patch in the configured corner."""
    img = _solid_image(128, 128)
    spec = AttackTransformConfig(
        name="badnet_oga",
        kind="badnet_patch",
        goal="oga",
        poison_negative=True,
        poison_positive=False,
        params={"patch_frac": 0.07, "position": "bottom_right"},
    )
    out = apply_attack_transform(img, spec, seed=7)
    diff = np.any(out != img, axis=-1)
    ys, xs = np.where(diff)
    assert diff.any(), "trigger had no effect on the image"
    cx, cy = float(xs.mean()), float(ys.mean())
    # Bottom-right quadrant of a 128x128 image.
    assert cx > 64.0 and cy > 64.0, f"OGA patch not in bottom-right ({cx},{cy})"


def test_semantic_green_applies_visible_change() -> None:
    img = _solid_image(size=96, value=160)
    spec = AttackTransformConfig(
        name="semantic_green_cleanlabel",
        kind="semantic_green",
        goal="semantic",
        params={"strength": 0.42},
    )
    out = apply_attack_transform(img, spec, seed=3)
    assert not np.array_equal(out, img)
    # When dispatched through attack_zoo the semantic polygon is authored in
    # RGB; the output must remain BGR to stay compatible with cv2 IO.
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_wanet_produces_warp_not_noise() -> None:
    # Use a non-uniform image so that a small warp is actually observable.
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    spec = AttackTransformConfig(
        name="wanet_oga",
        kind="wanet",
        goal="oga",
        poison_negative=True,
        poison_positive=False,
        params={"amplitude": 0.05, "grid": 5},
    )
    out = apply_attack_transform(img, spec, seed=7)
    assert not np.array_equal(out, img)
    assert out.shape == img.shape


def test_legacy_fallback_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """When attack_zoo is unavailable, the legacy NumPy helpers should run."""
    monkeypatch.setattr(mod, "_ATTACK_ZOO_AVAILABLE", False)
    monkeypatch.setattr(mod, "AttackSpec", None)
    monkeypatch.setattr(mod, "apply_attack_image", None)
    img = _solid_image(64, 64)
    spec = AttackTransformConfig(
        name="badnet_oga",
        kind="badnet_patch",
        goal="oga",
        params={"patch_frac": 0.12, "position": "br"},
    )
    out = apply_attack_transform(img, spec, seed=0)
    assert not np.array_equal(out, img)
