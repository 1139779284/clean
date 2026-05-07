from __future__ import annotations

import numpy as np

from model_security_gate.detox.yolo_dataset import _transform_normalized_labels_for_letterbox


def test_letterbox_label_transform_adds_vertical_padding() -> None:
    boxes = np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    transformed, keep = _transform_normalized_labels_for_letterbox(
        boxes,
        orig_shape=(100, 200),
        resized_shape=(416, 416),
        scale=(2.08, 2.08),
        pad=(0.0, 104.0),
    )

    assert keep.tolist() == [True]
    assert transformed.shape == (1, 4)
    np.testing.assert_allclose(transformed[0], [0.5, 0.5, 0.5, 0.25], atol=1e-6)


def test_letterbox_label_transform_handles_empty_boxes() -> None:
    transformed, keep = _transform_normalized_labels_for_letterbox(
        np.zeros((0, 4), dtype=np.float32),
        orig_shape=(100, 200),
        resized_shape=(416, 416),
        scale=(2.08, 2.08),
        pad=(0.0, 104.0),
    )

    assert transformed.shape == (0, 4)
    assert keep.shape == (0,)
