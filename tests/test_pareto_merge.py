from __future__ import annotations

import torch

from model_security_gate.detox.pareto_merge import (
    alpha_for_state_key,
    generate_group_layer_alpha_specs,
    interpolate_state_dicts,
    layer_index_for_state_key,
    parse_alpha_grid,
    parse_layer_alpha_spec,
    parse_named_layer_alpha_specs,
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


def test_parse_named_layer_alpha_specs() -> None:
    specs = parse_named_layer_alpha_specs("head_high::0-9:0.1,10-21:0.3,22-999:0.8|0-9:0.8,10-999:0.2")
    assert [s.name for s in specs] == ["head_high", "layer_2"]
    assert specs[0].alpha_by_layer["22-999"] == 0.8
    assert specs[1].alpha_by_layer["0-9"] == 0.8


def test_generate_group_layer_alpha_specs_prioritizes_layer_grafts() -> None:
    specs = generate_group_layer_alpha_specs([0.0, 0.5, 1.0], max_candidates=4)
    assert len(specs) == 4
    assert all({"0-9", "10-21", "22-999"} <= set(spec.alpha_by_layer) for spec in specs)
    # The generator should put high-spread layer grafts before all-mid/all-same candidates.
    assert max(specs[0].alpha_by_layer.values()) - min(specs[0].alpha_by_layer.values()) == 1.0


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
