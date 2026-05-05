from model_security_gate.detox.pseudo_labels import summarize_pseudo_label_quality


def test_pseudo_quality_summary_counts_rejections():
    rows = [
        {"accepted": True, "n_pseudo_boxes": 2, "mean_teacher_conf": 0.8, "mean_suspicious_conf": 0.7, "agreement_rate": 1.0, "empty_label": False},
        {"accepted": False, "n_pseudo_boxes": 0, "mean_teacher_conf": None, "mean_suspicious_conf": 0.5, "agreement_rate": 0.0, "empty_label": True, "rejected_reason": "teacher_empty"},
    ]
    summary = summarize_pseudo_label_quality(rows)
    assert summary["n_images"] == 2
    assert summary["n_accepted"] == 1
    assert summary["n_rejected"] == 1
    assert summary["n_pseudo_boxes"] == 2
    assert summary["rejection_reasons"]["teacher_empty"] == 1
