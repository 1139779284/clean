import json

import numpy as np

from model_security_gate.scan.abs import abs_channel_scores, detect_abs_suspicious_channels


def test_abs_channel_scores_rank_target_correlated_channel():
    activations = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
        ]
    )
    target_scores = np.array([0.0, 0.2, 0.8, 1.0])
    scores = abs_channel_scores(activations, target_scores)
    assert int(np.argmax(scores)) == 0


def test_abs_cli(tmp_path):
    inp = tmp_path / "acts.json"
    out = tmp_path / "abs.json"
    inp.write_text(
        json.dumps({"activations": [[0, 1], [1, 1], [2, 1]], "target_scores": [0.0, 0.5, 1.0]}),
        encoding="utf-8",
    )
    result = detect_abs_suspicious_channels([[0, 1], [1, 1], [2, 1]], [0.0, 0.5, 1.0], top_fraction=0.5)
    out.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["suspicious_channels"]
