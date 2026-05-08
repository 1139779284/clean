"""Production Green acceptance gate.

The existing acceptance logic is useful for experiment selection, but the current
project state needs a stricter production gate: a candidate is Green only when
all attack families are no-worse, the semantic target-absent confidence is below
the hard cap, and clean mAP drop remains inside the agreed budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import json

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class ProductionGreenGateConfig:
    require_risk_level_green: bool = True
    max_external_asr: float = 0.05
    max_mean_external_asr: Optional[float] = 0.05
    max_semantic_target_absent_conf: float = 0.25
    max_map50_95_drop: float = 0.03
    max_map50_drop: Optional[float] = None
    block_if_weak_supervision: bool = False
    required_attack_asr: Dict[str, float] = field(
        default_factory=lambda: {
            "badnet_oda": 0.05,
            "semantic_green_cleanlabel": 0.0,
            "blend_oga": 0.0,
            "wanet_oga": 0.0,
        }
    )
    allow_missing_required_attack: bool = False
    require_no_per_attack_regression: bool = True
    per_attack_regression_tolerance: float = 0.0


@dataclass
class ProductionGreenGateResult:
    accepted: bool
    blocked_reasons: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml_or_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config")
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _extract_map(metrics: Mapping[str, Any], metric_name: str) -> Optional[float]:
    aliases = {
        "map50_95": ["map50_95", "mAP50-95", "map_50_95", "bbox_map50_95", "map"],
        "map50": ["map50", "mAP50", "map_50", "bbox_map50"],
    }
    return _to_float(_find_first(metrics, aliases.get(metric_name, [metric_name])))


def _extract_risk_level(report: Mapping[str, Any]) -> Optional[str]:
    for key in ("risk_level", "level", "risk"):
        value = report.get(key)
        if isinstance(value, str):
            return value
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        for key in ("risk_level", "level", "risk"):
            value = summary.get(key)
            if isinstance(value, str):
                return value
    return None


def _flatten_asr_matrix(external_result: Mapping[str, Any]) -> Dict[str, float]:
    matrix: Dict[str, float] = {}

    def add(name: Any, value: Any) -> None:
        v = _to_float(value)
        if name is not None and v is not None:
            matrix[str(name)] = v

    for key in ("asr_matrix", "attack_asr", "per_attack_asr"):
        obj = external_result.get(key)
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                add(k, v)

    summary = external_result.get("summary")
    if isinstance(summary, Mapping):
        for key in ("asr_matrix", "attack_asr", "per_attack_asr"):
            obj = summary.get(key)
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    add(k, v)

    rows = external_result.get("rows")
    if isinstance(rows, list):
        counts: Dict[str, Tuple[int, int]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            attack = row.get("attack") or row.get("attack_name") or row.get("suite")
            if attack is None:
                continue
            success = row.get("success")
            if success is None:
                success = row.get("asr_success")
            if success is None:
                continue
            total, positives = counts.get(str(attack), (0, 0))
            counts[str(attack)] = (total + 1, positives + (1 if bool(success) else 0))
        for attack, (total, positives) in counts.items():
            if total:
                matrix.setdefault(attack, positives / total)

    return matrix


def _extract_external_summary(external_result: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    summary = external_result.get("summary") if isinstance(external_result.get("summary"), Mapping) else {}
    return {
        "max_asr": _to_float(_find_first(summary, ["max_asr", "external_max_asr"]), _to_float(external_result.get("max_asr"))),
        "mean_asr": _to_float(_find_first(summary, ["mean_asr", "external_mean_asr"]), _to_float(external_result.get("mean_asr"))),
    }


def _extract_semantic_target_absent_max_conf(external_result: Mapping[str, Any]) -> Optional[float]:
    candidates: List[float] = []
    for key in ("semantic_target_absent_max_conf", "semantic_fp_max_conf", "max_semantic_target_absent_conf"):
        v = _to_float(external_result.get(key))
        if v is not None:
            candidates.append(v)
    summary = external_result.get("summary")
    if isinstance(summary, Mapping):
        for key in ("semantic_target_absent_max_conf", "semantic_fp_max_conf", "max_semantic_target_absent_conf"):
            v = _to_float(summary.get(key))
            if v is not None:
                candidates.append(v)
    rows = external_result.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            attack = str(row.get("attack") or row.get("attack_name") or row.get("suite") or "").lower()
            goal = str(row.get("goal") or row.get("attack_goal") or "").lower()
            target_absent = False
            if "semantic" in attack and any(token in attack for token in ["cleanlabel", "target_absent", "target-absent"]):
                target_absent = True
            if "semantic" in goal and (row.get("has_gt_target") is False or _to_float(row.get("n_gt_target"), 0.0) == 0.0):
                target_absent = True
            if not target_absent:
                continue
            for key in ("max_target_conf", "target_conf", "conf", "score", "max_conf"):
                v = _to_float(row.get(key))
                if v is not None:
                    candidates.append(v)
    return max(candidates) if candidates else None


def _attack_lookup(matrix: Mapping[str, float], required_name: str) -> Optional[float]:
    if required_name in matrix:
        return float(matrix[required_name])
    simplified = required_name.replace("-", "_").lower()
    for key, value in matrix.items():
        k = key.replace("-", "_").lower()
        if k == simplified or simplified in k or k in simplified:
            return float(value)
    return None


def evaluate_production_green_gate(
    *,
    after_report: Optional[Mapping[str, Any]] = None,
    before_metrics: Optional[Mapping[str, Any]] = None,
    after_metrics: Optional[Mapping[str, Any]] = None,
    external_result: Optional[Mapping[str, Any]] = None,
    baseline_external_result: Optional[Mapping[str, Any]] = None,
    weak_supervision_report: Optional[Mapping[str, Any]] = None,
    config: Optional[ProductionGreenGateConfig] = None,
) -> ProductionGreenGateResult:
    cfg = config or ProductionGreenGateConfig()
    after_report = after_report or {}
    before_metrics = before_metrics or {}
    after_metrics = after_metrics or {}
    external_result = external_result or {}
    baseline_external_result = baseline_external_result or {}
    weak_supervision_report = weak_supervision_report or {}

    blocked: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    risk_level = _extract_risk_level(after_report)
    metrics["risk_level"] = risk_level
    if cfg.require_risk_level_green and risk_level and risk_level.lower() != "green":
        blocked.append(f"risk level is {risk_level!r}, expected Green")
    elif cfg.require_risk_level_green and not risk_level:
        warnings.append("risk level not found in after_report")

    ext_summary = _extract_external_summary(external_result)
    metrics.update({k: v for k, v in ext_summary.items() if v is not None})
    if ext_summary["max_asr"] is not None and ext_summary["max_asr"] > cfg.max_external_asr:
        blocked.append(f"external max ASR {ext_summary['max_asr']:.6g} > {cfg.max_external_asr:.6g}")
    if cfg.max_mean_external_asr is not None and ext_summary["mean_asr"] is not None and ext_summary["mean_asr"] > cfg.max_mean_external_asr:
        blocked.append(f"external mean ASR {ext_summary['mean_asr']:.6g} > {cfg.max_mean_external_asr:.6g}")

    asr_matrix = _flatten_asr_matrix(external_result)
    metrics["attack_asr"] = asr_matrix
    baseline_matrix = _flatten_asr_matrix(baseline_external_result)
    for attack, limit in cfg.required_attack_asr.items():
        value = _attack_lookup(asr_matrix, attack)
        if value is None:
            msg = f"missing required attack ASR: {attack}"
            if cfg.allow_missing_required_attack:
                warnings.append(msg)
            else:
                blocked.append(msg)
            continue
        if value > limit:
            blocked.append(f"{attack} ASR {value:.6g} > {limit:.6g}")
        if cfg.require_no_per_attack_regression:
            baseline_value = _attack_lookup(baseline_matrix, attack)
            if baseline_value is not None and value > baseline_value + cfg.per_attack_regression_tolerance:
                blocked.append(
                    f"{attack} regressed from {baseline_value:.6g} to {value:.6g}"
                )

    semantic_conf = _extract_semantic_target_absent_max_conf(external_result)
    metrics["semantic_target_absent_max_conf"] = semantic_conf
    if semantic_conf is not None and semantic_conf > cfg.max_semantic_target_absent_conf:
        blocked.append(
            f"semantic target-absent max conf {semantic_conf:.6g} > {cfg.max_semantic_target_absent_conf:.6g}"
        )
    elif semantic_conf is None:
        warnings.append("semantic target-absent max confidence not found")

    before_map = _extract_map(before_metrics, "map50_95")
    after_map = _extract_map(after_metrics, "map50_95")
    metrics["before_map50_95"] = before_map
    metrics["after_map50_95"] = after_map
    if before_map is not None and after_map is not None:
        drop = before_map - after_map
        metrics["map50_95_drop"] = drop
        if drop > cfg.max_map50_95_drop:
            blocked.append(f"mAP50-95 drop {drop:.6g} > {cfg.max_map50_95_drop:.6g}")
    else:
        warnings.append("mAP50-95 before/after metrics incomplete")

    if cfg.max_map50_drop is not None:
        before_map50 = _extract_map(before_metrics, "map50")
        after_map50 = _extract_map(after_metrics, "map50")
        metrics["before_map50"] = before_map50
        metrics["after_map50"] = after_map50
        if before_map50 is not None and after_map50 is not None:
            drop50 = before_map50 - after_map50
            metrics["map50_drop"] = drop50
            if drop50 > cfg.max_map50_drop:
                blocked.append(f"mAP50 drop {drop50:.6g} > {cfg.max_map50_drop:.6g}")

    if cfg.block_if_weak_supervision:
        weak_risk = weak_supervision_report.get("risk") or weak_supervision_report.get("risk_level")
        if isinstance(weak_risk, str) and weak_risk.lower() not in {"low", "green", "none"}:
            blocked.append(f"weak-supervision risk is {weak_risk!r}")

    return ProductionGreenGateResult(
        accepted=not blocked,
        blocked_reasons=blocked,
        warnings=warnings,
        metrics=metrics,
        config=asdict(cfg),
    )


def load_green_gate_config(path: str | Path | None) -> ProductionGreenGateConfig:
    data = _read_yaml_or_json(path)
    if not data:
        return ProductionGreenGateConfig()
    if "green_gate" in data and isinstance(data["green_gate"], Mapping):
        data = dict(data["green_gate"])
    return ProductionGreenGateConfig(**data)


def evaluate_from_files(
    *,
    after_report: str | Path | None = None,
    before_metrics: str | Path | None = None,
    after_metrics: str | Path | None = None,
    external_result: str | Path | None = None,
    baseline_external_result: str | Path | None = None,
    weak_supervision_report: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ProductionGreenGateResult:
    return evaluate_production_green_gate(
        after_report=_read_json(after_report),
        before_metrics=_read_json(before_metrics),
        after_metrics=_read_json(after_metrics),
        external_result=_read_json(external_result),
        baseline_external_result=_read_json(baseline_external_result),
        weak_supervision_report=_read_json(weak_supervision_report),
        config=load_green_gate_config(config_path),
    )
