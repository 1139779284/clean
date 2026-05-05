from pathlib import Path

import cv2
import numpy as np

from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig, ASRAwareDatasetConfig, apply_attack_transform, build_asr_aware_yolo_dataset
from model_security_gate.utils.io import read_yaml


def test_attack_transform_changes_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = apply_attack_transform(img, AttackTransformConfig("badnet", kind="badnet_patch", params={"patch_frac": 0.2}))
    assert out.shape == img.shape
    assert int(out.sum()) > 0


def test_build_asr_aware_dataset_keeps_labels(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    img = np.full((80, 80, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(images / "a.jpg"), img)
    (labels / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data_yaml = build_asr_aware_yolo_dataset(
        images,
        labels,
        tmp_path / "out",
        class_names={0: "helmet"},
        cfg=ASRAwareDatasetConfig(
            val_fraction=0.0,
            target_class_ids=[0],
            include_clean_repeat=1,
            include_attack_repeat=1,
            attacks=[AttackTransformConfig("badnet_oda", kind="badnet_patch", goal="oda", poison_negative=False, poison_positive=True)],
        ),
    )
    data = read_yaml(data_yaml)
    assert data["label_mode"] == "asr_aware_supervised"
    train_labels = list((tmp_path / "out" / "labels" / "train").glob("*.txt"))
    assert len(train_labels) == 2
    assert all("0 0.500000" in p.read_text(encoding="utf-8") for p in train_labels)
