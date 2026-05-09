from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class SpectralSignatureResult:
    scores: list[float]
    suspicious_indices: list[int]
    threshold: float
    top_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "suspicious_indices": self.suspicious_indices,
            "threshold": float(self.threshold),
            "top_fraction": float(self.top_fraction),
        }


def spectral_signature_scores(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    top = vt[0]
    proj = x @ top
    return proj * proj


def detect_spectral_outliers(features: Sequence[Sequence[float]] | np.ndarray, *, top_fraction: float = 0.15) -> SpectralSignatureResult:
    scores = spectral_signature_scores(features)
    n = int(scores.shape[0])
    if n == 0:
        return SpectralSignatureResult([], [], 0.0, float(top_fraction))
    k = max(1, int(round(n * float(top_fraction))))
    order = np.argsort(scores)[::-1]
    suspicious = sorted(int(i) for i in order[:k])
    threshold = float(scores[order[k - 1]])
    return SpectralSignatureResult([float(x) for x in scores], suspicious, threshold, float(top_fraction))
