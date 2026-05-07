from __future__ import annotations

import torch
import torch.nn.functional as F

from model_security_gate.detox.pgbd_od import make_pgbd_attack_view, pgbd_paired_displacement_loss
from model_security_gate.detox.prototype import PrototypeBank


def _bank() -> PrototypeBank:
    return PrototypeBank(layer_name="layer", dim=2, prototypes={0: F.normalize(torch.tensor([1.0, 0.0]), dim=0)}, counts={0: 1})


def test_make_pgbd_attack_view_changes_image() -> None:
    img = torch.full((1, 3, 16, 16), 0.3)
    view = make_pgbd_attack_view(img, mode="mixed")
    assert view.shape == img.shape
    assert not torch.allclose(view, img)


def test_pgbd_negative_displacement_penalizes_target_like_shift() -> None:
    clean = {"layer": torch.zeros((1, 2, 4, 4), dtype=torch.float32)}
    attacked = {"layer": torch.zeros((1, 2, 4, 4), dtype=torch.float32)}
    attacked["layer"][:, 0] = 2.0  # target-prototype-like global feature
    batch = {
        "img": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
    }
    loss = pgbd_paired_displacement_loss(clean, attacked, batch, _bank(), [0], negative_margin=0.1)
    assert float(loss) > 0.0


def test_pgbd_positive_pair_loss_penalizes_roi_shift() -> None:
    clean = {"layer": torch.zeros((1, 2, 4, 4), dtype=torch.float32)}
    attacked = {"layer": torch.zeros((1, 2, 4, 4), dtype=torch.float32)}
    clean["layer"][:, 0] = 1.0
    attacked["layer"][:, 1] = 1.0
    batch = {
        "img": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
    }
    loss = pgbd_paired_displacement_loss(clean, attacked, batch, _bank(), [0])
    assert float(loss) > 0.0
