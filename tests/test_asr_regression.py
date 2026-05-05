from pathlib import Path

import cv2
import numpy as np

from model_security_gate.adapters.base import Detection
from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig
from model_security_gate.detox.asr_regression import ASRRegressionConfig, run_asr_regression


class FakeAdapter:
    names = {0: "helmet"}

    def predict_image(self, image, conf=None, iou=None, imgsz=None):
        arr = image if not isinstance(image, (str, Path)) else cv2.imread(str(image))
        # badnet patch creates bright pixels in a black test image; treat as ghost.
        if int(arr.max()) > 200:
            return [Detection((10, 10, 30, 30), 0.9, 0, "helmet")]
        return []

    def predict_batch(self, images, conf=None, iou=None, imgsz=None):
        return [self.predict_image(x, conf=conf, iou=iou, imgsz=imgsz) for x in images]


def test_asr_regression_detects_oga_success(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "neg.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    (labels / "neg.txt").write_text("", encoding="utf-8")
    result = run_asr_regression(
        FakeAdapter(),
        images,
        labels,
        target_class_ids=[0],
        cfg=ASRRegressionConfig(attacks=[AttackTransformConfig("badnet_oga", kind="badnet_patch", goal="oga")]),
    )
    assert result["summary"]["max_asr"] == 1.0
    assert result["summary"]["asr_matrix"]["badnet_oga"] == 1.0
