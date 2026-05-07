from pathlib import Path

import cv2
import numpy as np

from model_security_gate.detox.targeted_repair import (
    TargetedRepairConfig,
    _copy_clean_anchors,
    _select_attack_names,
    _target_ids_from_names,
    select_final_repair_candidate,
)
from model_security_gate.utils.io import write_yaml


def _write_img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))


def test_target_ids_from_names_supports_strings_and_numbers(tmp_path: Path):
    data = tmp_path / "data.yaml"
    write_yaml(data, {"path": str(tmp_path), "names": ["helmet", "head"]})
    assert _target_ids_from_names(data, ["helmet", "1"]) == [0, 1]


def test_select_attack_names_by_goal():
    names = ["badnet_oda", "wanet_oga", "semantic_green_cleanlabel"]
    assert _select_attack_names(names, [], "oda") == ["badnet_oda"]
    assert _select_attack_names(names, [], "oga") == ["wanet_oga"]
    assert _select_attack_names(names, ["badnet_oda"], "all") == ["badnet_oda"]


def test_copy_clean_anchors_creates_yolo_pairs(tmp_path: Path):
    root = tmp_path / "clean"
    img = root / "images" / "train" / "a.jpg"
    lab = root / "labels" / "train" / "a.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data = root / "data.yaml"
    write_yaml(data, {"path": str(root), "train": "images/train", "val": "images/train", "names": ["helmet"]})

    out = tmp_path / "repair"
    stats = _copy_clean_anchors(out, data, max_images=1)

    assert stats["added"] == 1
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 1
    labels = list((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8").strip().startswith("0 ")


def test_targeted_repair_config_defaults_are_oda_focused():
    cfg = TargetedRepairConfig(model="m.pt", data_yaml="data.yaml", out_dir="runs/x", external_roots=["bench"], target_classes=["helmet"])
    assert cfg.repair_goal == "oda"
    assert cfg.lambda_oda_matched > cfg.lambda_task
    assert cfg.lambda_oga_negative == 0.0


def test_select_final_repair_candidate_rolls_back_when_all_blocked():
    selection = select_final_repair_candidate(
        [
            {"model": "bad.pt", "score": 0.1, "blocked_attacks": ["badnet_oda"], "accepted": False},
            {"model": "worse.pt", "score": 0.2, "blocked_attacks": ["wanet_oga"], "accepted": False},
        ],
        fallback_model="input.pt",
    )
    assert selection["rolled_back"] is True
    assert selection["final_model"] == "input.pt"
    assert selection["best"] is None
    assert selection["best_by_score"]["model"] == "bad.pt"


def test_select_final_repair_candidate_can_choose_unblocked_candidate():
    selection = select_final_repair_candidate(
        [
            {"model": "blocked.pt", "score": 0.1, "blocked_attacks": ["badnet_oda"], "accepted": False},
            {"model": "safe.pt", "score": 0.2, "blocked_attacks": [], "accepted": False},
        ],
        fallback_model="input.pt",
    )
    assert selection["rolled_back"] is False
    assert selection["final_model"] == "safe.pt"
    assert selection["best"]["model"] == "safe.pt"
