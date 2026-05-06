from __future__ import annotations

import torch

from model_security_gate.detox.pareto_merge import (
    alpha_for_state_key,
    interpolate_state_dicts,
    layer_index_for_state_key,
    parse_alpha_grid,
    parse_layer_alpha_spec,
)


def test_parse_alpha_grid() -> None:
    assert parse_alpha_grid("0,0.25,1") == [0.0, 0.25, 1.0]


def test_layer_index_for_yolo_key() -> None:
    assert layer_index_for_state_key("model.22.cv3.0.0.conv.weight") == 22
    assert layer_index_for_state_key("head.weight") is None


def test_layer_alpha_override() -> None:
    spec = parse_layer_alpha_spec("0-9:0.2,10-99:0.8")
    assert alpha_for_state_key("model.3.conv.weight", 0.5, spec) == 0.2
    assert alpha_for_state_key("model.22.conv.weight", 0.5, spec) == 0.8
    assert alpha_for_state_key("other.weight", 0.5, spec) == 0.5


def test_interpolate_state_dicts_keeps_non_float_buffers() -> None:
    base = {
        "model.0.weight": torch.tensor([0.0, 2.0]),
        "model.0.count": torch.tensor([1], dtype=torch.int64),
    }
    source = {
        "model.0.weight": torch.tensor([2.0, 4.0]),
        "model.0.count": torch.tensor([9], dtype=torch.int64),
    }
    merged, stats = interpolate_state_dicts(base, source, alpha=0.25)
    assert torch.allclose(merged["model.0.weight"], torch.tensor([0.5, 2.5]))
    assert torch.equal(merged["model.0.count"], torch.tensor([1], dtype=torch.int64))
    assert stats["tensors_merged"] == 1
    assert stats["tensors_non_float"] == 1
