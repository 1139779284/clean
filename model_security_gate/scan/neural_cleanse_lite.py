from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class NeuralCleanseLiteResult:
    anomaly_index: float
    median_mask_norm: float
    target_norms: dict[str, float]
    suspicious_targets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_index": float(self.anomaly_index),
            "median_mask_norm": float(self.median_mask_norm),
            "target_norms": {str(k): float(v) for k, v in self.target_norms.items()},
            "suspicious_targets": self.suspicious_targets,
        }


def neural_cleanse_anomaly_from_mask_norms(mask_norms: Mapping[str, float], *, threshold: float = 2.0) -> NeuralCleanseLiteResult:
    """Compute Neural-Cleanse-style MAD anomaly index from trigger mask norms.

    This module intentionally does not perform heavy trigger inversion in CI;
    it provides the robust statistic used after an inversion job has produced
    per-target mask norms.
    """

    vals = np.asarray([float(v) for v in mask_norms.values()], dtype=np.float64)
    if vals.size == 0:
        return NeuralCleanseLiteResult(0.0, 0.0, {}, [])
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median))) + 1e-12
    scores = {str(k): (median - float(v)) / mad for k, v in mask_norms.items()}
    anomaly = max([0.0] + list(scores.values()))
    suspicious = sorted([k for k, s in scores.items() if s >= float(threshold)])
    return NeuralCleanseLiteResult(float(anomaly), median, {str(k): float(v) for k, v in mask_norms.items()}, suspicious)
