from pathlib import Path

import cv2
import numpy as np

from model_security_gate.adapters.base import Detection
from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    discover_external_attack_datasets,
    run_external_hard_suite,
)


class AlwaysDetectAdapter:
    names = {0: "helmet"}

    def predict_image(self, image, conf=None, iou=None, imgsz=None):
        return [Detection((10, 10, 30, 30), 0.9, 0, "helmet")]

    def predict_batch(self, images, conf=None, iou=None, imgsz=None):
        return [self.predict_image(x, conf=conf, iou=iou, imgsz=imgsz) for x in images]


class NeverDetectAdapter:
    names = {0: "helmet"}

    def predict_image(self, image, conf=None, iou=None, imgsz=None):
        return []

    def predict_batch(self, images, conf=None, iou=None, imgsz=None):
        return [[] for _ in images]


class MatchDetectAdapter:
    names = {0: "helmet"}

    def predict_image(self, image, conf=None, iou=None, imgsz=None):
        return [Detection((20, 20, 45, 45), 0.9, 0, "helmet")]

    def predict_batch(self, images, conf=None, iou=None, imgsz=None):
        return [self.predict_image(x, conf=conf, iou=iou, imgsz=imgsz) for x in images]


def _write_img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))


def test_discover_and_score_oga_external_suite(tmp_path: Path):
    root = tmp_path / "bench"
    img = root / "data" / "badnet_oga" / "images" / "val" / "neg.jpg"
    lab = root / "data" / "badnet_oga" / "labels" / "val" / "neg.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("", encoding="utf-8")

    datasets = discover_external_attack_datasets([root])
    assert len(datasets) == 1
    assert datasets[0].goal == "oga"

    result = run_external_hard_suite(AlwaysDetectAdapter(), [0], ExternalHardSuiteConfig(attacks=datasets))
    assert result["summary"]["max_asr"] == 1.0
    assert result["summary"]["asr_matrix"]["bench::badnet_oga"] == 1.0


def test_score_oda_external_suite(tmp_path: Path):
    root = tmp_path / "bench"
    img = root / "data" / "badnet_oda" / "images" / "val" / "pos.jpg"
    lab = root / "data" / "badnet_oda" / "labels" / "val" / "pos.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    datasets = discover_external_attack_datasets([root])
    result = run_external_hard_suite(NeverDetectAdapter(), [0], ExternalHardSuiteConfig(attacks=datasets))
    assert result["summary"]["max_asr"] == 1.0


def test_oda_rows_explain_unmatched_target_detections(tmp_path: Path):
    root = tmp_path / "bench"
    img = root / "data" / "badnet_oda" / "images" / "val" / "pos.jpg"
    lab = root / "data" / "badnet_oda" / "labels" / "val" / "pos.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    datasets = discover_external_attack_datasets([root])

    result = run_external_hard_suite(AlwaysDetectAdapter(), [0], ExternalHardSuiteConfig(attacks=datasets, match_iou=0.5))
    row = result["rows"][0]

    assert row["success"] is True
    assert row["success_reason"] == "no_gt_target_recalled"
    assert row["oda_success_mode"] == "localized_any_recalled"
    assert row["n_gt_target"] == 1
    assert row["n_target_dets"] == 1
    assert row["n_recalled_target"] == 0
    assert row["best_target_iou"] < 0.5


def test_oda_matching_target_detection_is_not_success(tmp_path: Path):
    root = tmp_path / "bench"
    img = root / "data" / "badnet_oda" / "images" / "val" / "pos.jpg"
    lab = root / "data" / "badnet_oda" / "labels" / "val" / "pos.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    datasets = discover_external_attack_datasets([root])

    result = run_external_hard_suite(MatchDetectAdapter(), [0], ExternalHardSuiteConfig(attacks=datasets, match_iou=0.3))
    row = result["rows"][0]

    assert row["success"] is False
    assert row["success_reason"] == "at_least_one_gt_target_recalled"
    assert row["n_recalled_target"] == 1
    assert row["best_target_iou"] >= 0.3


def test_oda_class_presence_mode_counts_any_target_detection_as_not_success(tmp_path: Path):
    root = tmp_path / "bench"
    img = root / "data" / "badnet_oda" / "images" / "val" / "pos.jpg"
    lab = root / "data" / "badnet_oda" / "labels" / "val" / "pos.txt"
    _write_img(img)
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    datasets = discover_external_attack_datasets([root])

    result = run_external_hard_suite(
        AlwaysDetectAdapter(),
        [0],
        ExternalHardSuiteConfig(attacks=datasets, match_iou=0.5, oda_success_mode="class_presence"),
    )
    row = result["rows"][0]

    assert row["success"] is False
    assert row["success_reason"] == "target_class_still_detected"
    assert row["n_target_dets"] == 1


def test_failure_only_replay_copies_only_success_rows(tmp_path: Path):
    root = tmp_path / "bench"
    failed_img = root / "data" / "badnet_oga" / "images" / "val" / "failed.jpg"
    passed_img = root / "data" / "badnet_oga" / "images" / "val" / "passed.jpg"
    failed_lab = root / "data" / "badnet_oga" / "labels" / "val" / "failed.txt"
    passed_lab = root / "data" / "badnet_oga" / "labels" / "val" / "passed.txt"
    _write_img(failed_img)
    _write_img(passed_img)
    failed_lab.parent.mkdir(parents=True, exist_ok=True)
    failed_lab.write_text("", encoding="utf-8")
    passed_lab.write_text("", encoding="utf-8")

    datasets = discover_external_attack_datasets([root])
    out = tmp_path / "detox_ds"
    stats = append_external_replay_samples(
        output_dataset_dir=out,
        attack_datasets=datasets,
        target_class_ids=[0],
        selected_attack_names=["badnet_oga"],
        failure_rows=[
            {"image": str(failed_img), "attack": "badnet_oga", "success": True},
            {"image": str(passed_img), "attack": "badnet_oga", "success": False},
        ],
        failure_only=True,
        repeat=2,
    )

    copied = sorted((out / "images" / "train").glob("*.jpg"))
    assert stats["added"] == 2
    assert stats["repeat"] == 2
    assert len(copied) == 2
    assert "failed" in copied[0].name


def test_failure_only_replay_can_match_by_basename_across_roots(tmp_path: Path):
    replay_root = tmp_path / "replay"
    eval_root = tmp_path / "eval"
    replay_img = replay_root / "data" / "badnet_oga" / "images" / "val" / "shared.jpg"
    replay_lab = replay_root / "data" / "badnet_oga" / "labels" / "val" / "shared.txt"
    eval_img = eval_root / "data" / "badnet_oga" / "images" / "val" / "shared.jpg"
    _write_img(replay_img)
    replay_lab.parent.mkdir(parents=True, exist_ok=True)
    replay_lab.write_text("", encoding="utf-8")

    datasets = discover_external_attack_datasets([replay_root])
    out = tmp_path / "detox_ds"
    stats = append_external_replay_samples(
        output_dataset_dir=out,
        attack_datasets=datasets,
        target_class_ids=[0],
        selected_attack_names=["badnet_oga"],
        failure_rows=[{"image": str(eval_img), "attack": "badnet_oga", "success": True}],
        failure_only=True,
    )

    copied = sorted((out / "images" / "train").glob("*.jpg"))
    assert stats["added"] == 1
    assert len(copied) == 1
    assert "shared" in copied[0].name
