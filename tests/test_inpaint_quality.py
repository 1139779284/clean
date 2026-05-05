import cv2
import numpy as np

from model_security_gate.cf.transforms import assess_inpaint_quality
from model_security_gate.detox.dataset_builder import DetoxDatasetConfig, build_counterfactual_yolo_dataset


def test_inpaint_quality_rejects_huge_target_mask():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    report = assess_inpaint_quality(img, img.copy(), boxes=[(2, 2, 98, 98)], expand=0.0)
    assert report["accepted"] is False
    assert "mask_too_large" in report["reasons"]


def test_dataset_builder_skips_failed_target_inpaint(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(images / "sample.jpg"), img)
    (labels / "sample.txt").write_text("0 0.5 0.5 0.95 0.95\n", encoding="utf-8")

    out = tmp_path / "detox"
    build_counterfactual_yolo_dataset(
        images,
        labels,
        out,
        class_names=["helmet"],
        target_class_ids=[0],
        cfg=DetoxDatasetConfig(val_fraction=0.0, variants=["target_inpaint"], skip_failed_inpaint=True),
    )

    assert (out / "images" / "train" / "sample_orig.jpg").exists()
    assert not (out / "images" / "train" / "sample_target_inpaint.jpg").exists()
    manifest = (out / "counterfactual_quality_manifest.json").read_text(encoding="utf-8")
    assert '"skipped": 1' in manifest
