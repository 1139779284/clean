from pathlib import Path

import yaml
import torch

from model_security_gate.detox.hybrid_purify_train import _same_root_sets, _torch_device_arg
from model_security_gate.detox.strong_train import (
    _normalize_ultralytics_loss_args,
    _reset_ultralytics_criterion,
    _torch_model,
    _unfreeze_trainable_model,
)


def test_hybrid_purify_config_has_required_sections():
    cfg_path = Path("configs/hybrid_purify_detox.yaml")
    assert cfg_path.exists()
    data = yaml.safe_load(cfg_path.read_text())
    cfg = data["hybrid_purify_detox"]
    assert cfg["max_allowed_external_asr"] <= 0.10
    attacks = cfg["attacks"]
    names = {a["name"] for a in attacks}
    assert {"badnet_oga", "blend_oga", "wanet_oga", "badnet_oda", "semantic_green_cleanlabel"}.issubset(names)
    for a in attacks:
        if a["goal"] == "oga":
            assert a.get("poison_negative") is True
            assert a.get("poison_positive") is False
        if a["goal"] == "oda":
            assert a.get("poison_positive") is True
            assert a.get("poison_negative") is False


def test_hybrid_replay_failure_only_requires_same_roots(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    assert _same_root_sets([str(root_a)], [str(root_a.resolve())])
    assert not _same_root_sets([str(root_a)], [str(root_b)])


def test_hybrid_feature_purifier_normalizes_numeric_cuda_device():
    assert _torch_device_arg(0) == "cuda:0"
    assert _torch_device_arg("0") == "cuda:0"
    assert _torch_device_arg("cuda:1") == "cuda:1"


def test_strong_train_prefers_ultralytics_inner_model():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)

    wrapper = Wrapper()
    assert _torch_model(wrapper) is wrapper.model


def test_strong_train_normalizes_checkpoint_args_for_loss():
    model = torch.nn.Linear(1, 1)
    model.args = {"task": "detect", "imgsz": 640}

    _normalize_ultralytics_loss_args(model)

    assert hasattr(model.args, "box")
    assert hasattr(model.args, "cls")
    assert hasattr(model.args, "dfl")
    assert model.args.task == "detect"


def test_strong_train_unfreezes_loaded_student_weights():
    model = torch.nn.Linear(1, 1)
    for param in model.parameters():
        param.requires_grad_(False)

    _unfreeze_trainable_model(model)

    assert all(param.requires_grad for param in model.parameters())


def test_strong_train_resets_serialized_criterion():
    model = torch.nn.Linear(1, 1)
    model.criterion = object()

    _reset_ultralytics_criterion(model)

    assert model.criterion is None
