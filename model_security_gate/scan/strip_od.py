from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class StripODResult:
    entropy: float
    target_consistency: float
    mean_detection_count: float
    suspicious: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy": float(self.entropy),
            "target_consistency": float(self.target_consistency),
            "mean_detection_count": float(self.mean_detection_count),
            "suspicious": bool(self.suspicious),
            "reason": self.reason,
        }


def _class_id(det: Any) -> int | None:
    if isinstance(det, Mapping):
        v = det.get("cls_id", det.get("class_id", det.get("class")))
    else:
        v = getattr(det, "cls_id", None)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def strip_od_score(
    perturbation_detections: Sequence[Sequence[Any]],
    *,
    target_class_ids: Sequence[int],
    entropy_threshold: float = 0.35,
    target_consistency_threshold: float = 0.85,
) -> StripODResult:
    """Detector-adapted STRIP statistic over already-computed predictions.

    A sample is suspicious when target detections persist across many mixed or
    corrupted variants while the class distribution entropy remains very low.
    """

    target = {int(x) for x in target_class_ids}
    counts: dict[int, int] = {}
    target_hits = 0
    total_dets = 0
    for dets in perturbation_detections:
        has_target = False
        total_dets += len(dets)
        for det in dets:
            cid = _class_id(det)
            if cid is None:
                continue
            counts[cid] = counts.get(cid, 0) + 1
            if cid in target:
                has_target = True
        target_hits += int(has_target)
    n = max(1, len(perturbation_detections))
    total = max(1, sum(counts.values()))
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)
    if len(counts) > 1:
        entropy /= math.log(len(counts))
    consistency = target_hits / n
    mean_count = total_dets / n
    suspicious = entropy <= float(entropy_threshold) and consistency >= float(target_consistency_threshold)
    reason = "low_entropy_persistent_target" if suspicious else "not_suspicious"
    return StripODResult(entropy, consistency, mean_count, suspicious, reason)
