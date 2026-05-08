"""Runtime abstain rules for known semantic target-absent FP patterns.

This is a deployment safety net, not a substitute for model detox. It lets the
runtime guard route images to review when a known target-absent semantic failure
pattern appears again, for example a helmet false positive in a localized region
with score above a hard cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


BBox = Tuple[float, float, float, float]


@dataclass
class SemanticAbstainRule:
    rule_id: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    min_conf: float = 0.25
    region_xyxy: Optional[List[float]] = None
    min_region_iou: float = 0.0
    require_center_in_region: bool = False
    image_globs: List[str] = field(default_factory=list)
    reason: str = "known semantic target-absent FP pattern"
    enabled: bool = True


@dataclass
class SemanticAbstainDecision:
    action: str
    matched_rules: List[Dict[str, Any]]
    max_matched_conf: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bbox_from_detection(det: Mapping[str, Any]) -> Optional[BBox]:
    for key in ("xyxy", "bbox", "box"):
        value = det.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
            except (TypeError, ValueError):
                return None
    keys = ("x1", "y1", "x2", "y2")
    if all(k in det for k in keys):
        try:
            return tuple(float(det[k]) for k in keys)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _center_in_region(box: BBox, region: BBox) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = region
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _det_class_id(det: Mapping[str, Any]) -> Optional[int]:
    for key in ("class_id", "cls", "class"):
        value = det.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _det_class_name(det: Mapping[str, Any]) -> Optional[str]:
    for key in ("class_name", "name", "label"):
        value = det.get(key)
        if isinstance(value, str):
            return value
    return None


def _det_conf(det: Mapping[str, Any]) -> float:
    for key in ("conf", "confidence", "score", "prob"):
        value = det.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _image_matches(rule: SemanticAbstainRule, image_path: Optional[str]) -> bool:
    if not rule.image_globs:
        return True
    if not image_path:
        return False
    name = Path(image_path).name
    full = str(image_path)
    return any(fnmatch(name, pattern) or fnmatch(full, pattern) for pattern in rule.image_globs)


def detection_matches_rule(det: Mapping[str, Any], rule: SemanticAbstainRule, image_path: Optional[str] = None) -> bool:
    if not rule.enabled:
        return False
    if not _image_matches(rule, image_path):
        return False
    det_class_id = _det_class_id(det)
    det_class_name = _det_class_name(det)
    if rule.class_id is not None and det_class_id != rule.class_id:
        return False
    if rule.class_name is not None and (det_class_name or "").lower() != rule.class_name.lower():
        return False
    conf = _det_conf(det)
    if conf < rule.min_conf:
        return False
    if rule.region_xyxy is not None:
        if len(rule.region_xyxy) != 4:
            return False
        box = _bbox_from_detection(det)
        if box is None:
            return False
        region = tuple(float(x) for x in rule.region_xyxy)  # type: ignore[assignment]
        if rule.require_center_in_region and not _center_in_region(box, region):
            return False
        if rule.min_region_iou > 0 and _bbox_iou(box, region) < rule.min_region_iou:
            return False
    return True


def decide_semantic_abstain(
    detections: Sequence[Mapping[str, Any]],
    rules: Sequence[SemanticAbstainRule],
    *,
    image_path: Optional[str] = None,
) -> SemanticAbstainDecision:
    matches: List[Dict[str, Any]] = []
    max_conf: Optional[float] = None
    for det in detections:
        for rule in rules:
            if detection_matches_rule(det, rule, image_path=image_path):
                conf = _det_conf(det)
                max_conf = conf if max_conf is None else max(max_conf, conf)
                matches.append(
                    {
                        "rule_id": rule.rule_id,
                        "reason": rule.reason,
                        "class_id": _det_class_id(det),
                        "class_name": _det_class_name(det),
                        "conf": conf,
                        "bbox": _bbox_from_detection(det),
                    }
                )
    return SemanticAbstainDecision(action="review" if matches else "pass", matched_rules=matches, max_matched_conf=max_conf)


def load_semantic_abstain_rules(path: str | Path) -> List[SemanticAbstainRule]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML abstain rules")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if isinstance(data, Mapping):
        rules_data = data.get("semantic_abstain_rules", data.get("rules", []))
    else:
        rules_data = data
    if not isinstance(rules_data, list):
        raise ValueError("Expected a list of semantic abstain rules")
    return [SemanticAbstainRule(**dict(item)) for item in rules_data]
