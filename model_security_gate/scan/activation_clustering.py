from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ActivationClusteringResult:
    labels: list[int]
    distances: list[float]
    small_cluster: int | None
    suspicious_indices: list[int]
    cluster_sizes: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "distances": self.distances,
            "small_cluster": self.small_cluster,
            "suspicious_indices": self.suspicious_indices,
            "cluster_sizes": {str(k): int(v) for k, v in self.cluster_sizes.items()},
        }


def _pca_reduce(x: np.ndarray, n_components: int = 10) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] <= 1:
        return x.astype(np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(int(n_components), vt.shape[0])
    return xc @ vt[:k].T


def _two_means(x: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, x.shape[1] if x.ndim == 2 else 0))
    if n == 1:
        return np.zeros((1,), dtype=np.int64), x[:1].copy()
    # Deterministic farthest-point initialization.
    mean = x.mean(axis=0)
    i0 = int(np.argmax(np.linalg.norm(x - mean, axis=1)))
    i1 = int(np.argmax(np.linalg.norm(x - x[i0], axis=1)))
    centers = np.vstack([x[i0], x[i1]]).astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in [0, 1]:
            if np.any(labels == c):
                centers[c] = x[labels == c].mean(axis=0)
    return labels, centers


def activation_clustering(features: Sequence[Sequence[float]] | np.ndarray, *, pca_components: int = 10, small_cluster_fraction: float = 0.35) -> ActivationClusteringResult:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return ActivationClusteringResult([], [], None, [], {})
    z = _pca_reduce(x, pca_components)
    labels, centers = _two_means(z)
    sizes = {int(c): int(np.sum(labels == c)) for c in sorted(set(int(v) for v in labels.tolist()))}
    small_cluster = None
    suspicious: list[int] = []
    if len(sizes) >= 2:
        c_min = min(sizes, key=sizes.get)
        if sizes[c_min] / max(1, x.shape[0]) <= float(small_cluster_fraction):
            small_cluster = int(c_min)
            suspicious = [int(i) for i, lab in enumerate(labels) if int(lab) == small_cluster]
    d = np.linalg.norm(z - centers[labels], axis=1) if centers.size else np.zeros((x.shape[0],), dtype=np.float64)
    return ActivationClusteringResult([int(v) for v in labels], [float(v) for v in d], small_cluster, suspicious, sizes)
