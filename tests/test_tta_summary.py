import pandas as pd

from model_security_gate.scan.tta_scan import summarize_tta


def test_tta_summary_does_not_flag_true_target_photometric_wobble() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "grayscale",
                "base_conf": 0.80,
                "variant_conf": 0.35,
                "conf_drop": 1.0 - 0.35 / 0.80,
                "matched_iou": 0.80,
                "eval_conf": 0.25,
                "has_gt_target": True,
                "base_overlaps_gt_target": True,
                "context_dependence": False,
                "target_removal_failure": False,
            }
        ]
    )

    summary = summarize_tta(df)

    assert summary["semantic_shortcut_rate"] == 0.0
    assert summary["context_color_dependency_rate"] == 0.0


def test_tta_summary_flags_color_vanish_below_threshold() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "low_saturation",
                "base_conf": 0.80,
                "variant_conf": 0.10,
                "conf_drop": 1.0 - 0.10 / 0.80,
                "matched_iou": 0.10,
                "eval_conf": 0.25,
                "has_gt_target": True,
                "base_overlaps_gt_target": True,
                "context_dependence": False,
                "target_removal_failure": False,
            }
        ]
    )

    summary = summarize_tta(df)

    assert summary["context_color_dependency_rate"] == 1.0


def test_tta_summary_flags_target_absent_semantic_fp() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "jpeg",
                "base_conf": 0.70,
                "variant_conf": 0.60,
                "conf_drop": 1.0 - 0.60 / 0.70,
                "matched_iou": 0.70,
                "eval_conf": 0.25,
                "has_gt_target": False,
                "base_overlaps_gt_target": False,
                "context_dependence": False,
                "target_removal_failure": False,
            }
        ]
    )

    summary = summarize_tta(df)

    assert summary["semantic_shortcut_rate"] == 1.0


def test_tta_summary_does_not_treat_extra_box_on_positive_image_as_semantic_shortcut() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "jpeg",
                "base_conf": 0.70,
                "variant_conf": 0.60,
                "conf_drop": 1.0 - 0.60 / 0.70,
                "matched_iou": 0.70,
                "eval_conf": 0.25,
                "has_gt_target": True,
                "base_overlaps_gt_target": False,
                "context_dependence": False,
                "target_removal_failure": False,
            }
        ]
    )

    summary = summarize_tta(df)

    assert summary["semantic_shortcut_rate"] == 0.0
