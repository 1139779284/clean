import numpy as np

from model_security_gate.scan.activation_clustering import activation_clustering
from model_security_gate.scan.neural_cleanse_lite import neural_cleanse_anomaly_from_mask_norms
from model_security_gate.scan.spectral_signatures import detect_spectral_outliers
from model_security_gate.scan.strip_od import strip_od_score


def test_spectral_signatures_flags_outlier():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    result = detect_spectral_outliers(x, top_fraction=0.25)
    assert result.suspicious_indices == [3]


def test_activation_clustering_small_cluster():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [4.0, 4.0]])
    result = activation_clustering(x, small_cluster_fraction=0.30)
    assert result.small_cluster is not None
    assert len(result.suspicious_indices) == 1


def test_strip_od_detects_persistent_low_entropy_target():
    dets = [[{"class_id": 0}], [{"class_id": 0}], [{"class_id": 0}], [{"class_id": 0}]]
    result = strip_od_score(dets, target_class_ids=[0])
    assert result.suspicious is True


def test_neural_cleanse_anomaly_statistic():
    result = neural_cleanse_anomaly_from_mask_norms({"helmet": 0.1, "head": 1.0, "person": 1.1}, threshold=2.0)
    assert "helmet" in result.suspicious_targets
