from __future__ import annotations

from model_security_gate.detox.oda_candidate_diagnostics import _best_recall_stats, _truthy, summarize_diagnostic_rows


def test_truthy_accepts_common_csv_values() -> None:
    assert _truthy(True) is True
    assert _truthy("true") is True
    assert _truthy("1") is True
    assert _truthy("yes") is True
    assert _truthy("false") is False


def test_best_recall_stats_matches_target_box() -> None:
    dets = [{"xyxy": [10.0, 10.0, 30.0, 30.0], "conf": 0.72}]
    labels = [{"xyxy": [12.0, 12.0, 31.0, 31.0], "cls_id": 0}]

    stats = _best_recall_stats(dets, labels, match_iou=0.30)

    assert stats["n_gt_target"] == 1
    assert stats["n_recalled_target"] == 1
    assert stats["best_conf"] == 0.72
    assert stats["best_iou"] > 0.60


def test_summarize_diagnostic_rows_identifies_raw_candidate_gap() -> None:
    rows = [
        {
            "lowconf_n_recalled_target": 0,
            "raw_near_gt_n_candidates": 17,
            "raw_near_gt_n_over_conf": 0,
            "raw_near_gt_best_target_score": 0.08,
        },
        {
            "lowconf_n_recalled_target": 1,
            "raw_near_gt_n_candidates": 24,
            "raw_near_gt_n_over_conf": 1,
            "raw_near_gt_best_target_score": 0.42,
        },
    ]

    summary = summarize_diagnostic_rows(rows, conf=0.25)

    assert summary["n"] == 2
    assert summary["lowconf_recalled_rate"] == 0.5
    assert summary["raw_any_near_gt_rate"] == 1.0
    assert summary["raw_near_gt_over_conf_rate"] == 0.5
    assert summary["raw_near_gt_best_target_score_mean"] == 0.25
