from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ABSResult:
    channel_scores: list[float]
    suspicious_channels: list[int]
    threshold: float
    top_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_scores": [float(x) for x in self.channel_scores],
            "suspicious_channels": [int(x) for x in self.suspicious_channels],
            "threshold": float(self.threshold),
            "top_fraction": float(self.top_fraction),
        }


def abs_channel_scores(
    activations: Sequence[Sequence[float]] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute a lightweight ABS-style channel stimulation score.

    Full ABS stimulates neurons and measures target-label dominance.  This
    lightweight scorer is the deterministic CI-safe statistic used after a hook
    job has collected per-sample channel activations and target scores.
    """

    x = np.asarray(activations, dtype=np.float64)
    y = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0 or y.shape[0] != x.shape[0]:
        return np.zeros((0,), dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denom = np.linalg.norm(x, axis=0) * (np.linalg.norm(y) + 1e-12)
    corr = np.divide(x.T @ y, denom + 1e-12)
    high = np.percentile(np.asarray(activations, dtype=np.float64), 90, axis=0)
    med = np.percentile(np.asarray(activations, dtype=np.float64), 50, axis=0)
    gain = np.maximum(0.0, high - med)
    if np.max(gain) > 0:
        gain = gain / (np.max(gain) + 1e-12)
    return np.maximum(0.0, corr) * (1.0 + gain)


def detect_abs_suspicious_channels(
    activations: Sequence[Sequence[float]] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
    *,
    top_fraction: float = 0.05,
) -> ABSResult:
    scores = abs_channel_scores(activations, target_scores)
    n = int(scores.shape[0])
    if n == 0:
        return ABSResult([], [], 0.0, float(top_fraction))
    k = max(1, int(round(n * float(top_fraction))))
    order = np.argsort(scores)[::-1]
    selected = sorted(int(i) for i in order[:k])
    threshold = float(scores[order[k - 1]])
    return ABSResult([float(x) for x in scores], selected, threshold, float(top_fraction))
