from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

RISK_ORDER = {"Green": 0, "Yellow": 1, "Red": 2, "Black": 3}
RISK_ORDER_LOWER = {k.lower(): v for k, v in RISK_ORDER.items()}


def _load_json_or_dict(obj: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    path = Path(obj)
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(report: Mapping[str, Any]) -> Dict[str, Any]:
    dec = report.get("decision", {})
    if isinstance(dec, str):
        return {"level": dec, "score": None, "reasons": []}
    if isinstance(dec, Mapping):
        return {
            "level": str(dec.get("level", "Unknown")),
            "score": dec.get("score"),
            "reasons": list(dec.get("reasons", []) or []),
        }
    return {"level": "Unknown", "score": None, "reasons": []}


def _risk_rank(level: str | None) -> int:
    if level is None:
        return 999
    return RISK_ORDER_LOWER.get(str(level).lower(), 999)


def _summary_value(report: Mapping[str, Any], section: str, key: str, default: float = 0.0) -> float:
    try:
        return float(((report.get("summaries") or {}).get(section) or {}).get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


def _safe_reduction(before: float, after: float) -> float:
    before = float(before or 0.0)
    after = float(after or 0.0)
    if before <= 1e-12:
        return 1.0 if after <= 1e-12 else -1.0
    return (before - after) / before


def _extract_security_signals(report: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "slice_anomaly_rate": _summary_value(report, "slice", "slice_anomaly_rate"),
        "global_false_positive_rate": _summary_value(report, "slice", "global_false_positive_rate"),
        "global_false_negative_rate": _summary_value(report, "slice", "global_false_negative_rate"),
        "context_dependence_rate": _summary_value(report, "tta", "context_dependence_rate"),
        "target_removal_failure_rate": _summary_value(report, "tta", "target_removal_failure_rate"),
        "semantic_shortcut_rate": _summary_value(report, "tta", "semantic_shortcut_rate"),
        "context_color_dependency_rate": _summary_value(report, "tta", "context_color_dependency_rate"),
        "stress_target_bias_rate": _summary_value(report, "stress", "stress_target_bias_rate"),
        "stress_target_vanish_rate": _summary_value(report, "stress", "stress_target_vanish_rate"),
        "deformation_instability_rate": _summary_value(report, "stress", "deformation_instability_rate"),
        "wrong_region_attention_rate": _summary_value(report, "occlusion", "wrong_region_attention_rate"),
    }


def _fp_proxy(signals: Mapping[str, float]) -> float:
    """Conservative false-positive/backdoor proxy used for acceptance.

    One persistent failure mode is enough to block safety-critical deployment, so
    this is intentionally a max over major signals rather than an average.
    """
    return max(float(v or 0.0) for v in signals.values()) if signals else 0.0


def compare_security_reports(before_json: str | Path | Mapping[str, Any], after_json: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    """Compare two security_gate.py JSON reports."""
    before = _load_json_or_dict(before_json)
    after = _load_json_or_dict(after_json)
    before_dec = _decision(before)
    after_dec = _decision(after)
    before_signals = _extract_security_signals(before)
    after_signals = _extract_security_signals(after)
    signal_compare: Dict[str, Dict[str, float]] = {}
    for key in sorted(set(before_signals) | set(after_signals)):
        b = float(before_signals.get(key, 0.0) or 0.0)
        a = float(after_signals.get(key, 0.0) or 0.0)
        signal_compare[key] = {"before": b, "after": a, "delta": a - b, "reduction": _safe_reduction(b, a)}

    score_before = before_dec.get("score")
    score_after = after_dec.get("score")
    try:
        score_delta = float(score_after) - float(score_before) if score_before is not None and score_after is not None else None
    except (TypeError, ValueError):
        score_delta = None

    fp_before = _fp_proxy(before_signals)
    fp_after = _fp_proxy(after_signals)
    return {
        "risk_before": before_dec["level"],
        "risk_after": after_dec["level"],
        "risk_rank_before": _risk_rank(before_dec["level"]),
        "risk_rank_after": _risk_rank(after_dec["level"]),
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": score_delta,
        "risk_improved": _risk_rank(after_dec["level"]) < _risk_rank(before_dec["level"]),
        "risk_not_worse": _risk_rank(after_dec["level"]) <= _risk_rank(before_dec["level"]),
        "signals": signal_compare,
        "fp_proxy_before": fp_before,
        "fp_proxy_after": fp_after,
        "fp_proxy_reduction": _safe_reduction(fp_before, fp_after),
    }


def _metric(metrics: Mapping[str, Any] | None, *keys: str) -> float | None:
    if not metrics:
        return None
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None
    return None


def compare_yolo_metrics(before_metrics: Mapping[str, Any] | None, after_metrics: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Compare clean validation metrics from eval_yolo_metrics.py."""
    before_metrics = dict(before_metrics or {})
    after_metrics = dict(after_metrics or {})
    pairs = {
        "map50": ("map50",),
        "map50_95": ("map50_95", "map"),
        "precision": ("precision", "mp"),
        "recall": ("recall", "mr"),
    }
    out: Dict[str, Any] = {"available": bool(before_metrics and after_metrics), "metrics": {}}
    for public_key, keys in pairs.items():
        b = _metric(before_metrics, *keys)
        a = _metric(after_metrics, *keys)
        drop = None if b is None or a is None else b - a
        out["metrics"][public_key] = {"before": b, "after": a, "drop": drop, "delta": None if drop is None else -drop}
    out["map_drop"] = out["metrics"].get("map50_95", {}).get("drop")
    out["map50_drop"] = out["metrics"].get("map50", {}).get("drop")
    out["precision_drop"] = out["metrics"].get("precision", {}).get("drop")
    out["recall_drop"] = out["metrics"].get("recall", {}).get("drop")
    return out


def summarize_supervision_risk(detox_manifest: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return weak-supervision flags from a strong detox manifest."""
    if not detox_manifest:
        return {"available": False, "weak_supervision": False, "reason": ""}
    supervision = detox_manifest.get("supervision") or {}
    weak = bool(supervision.get("weak_supervision", False))
    reason = str(supervision.get("weak_reason") or "")
    label_mode = str(supervision.get("label_mode") or detox_manifest.get("label_mode") or "")
    stages = detox_manifest.get("stages") or []
    if label_mode == "feature_only":
        weak = True
        reason = reason or "feature_only mode is risk-reduction only"
    if any((stage or {}).get("name") == "fallback_suspicious_as_teacher" for stage in stages if isinstance(stage, Mapping)):
        weak = True
        reason = reason or "self-pseudo mode used the suspicious model as teacher"
    return {
        "available": True,
        "weak_supervision": weak,
        "reason": reason,
        "label_mode": label_mode,
        "verification_status": detox_manifest.get("verification_status"),
    }


def _walk_asr(obj: Any, path: str = "") -> Iterable[tuple[str, float]]:
    """Yield ASR-like scalar values from nested benchmark outputs.

    Supports common shapes such as ``{"max_asr": 0.9}``, ASR matrices, and
    per-attack rows with ``asr``/``attack_success_rate`` keys.
    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_s = str(key)
            next_path = f"{path}.{key_s}" if path else key_s
            lk = key_s.lower()
            in_asr_matrix = "asr_matrix" in next_path.lower()
            if isinstance(value, (int, float)) and (in_asr_matrix or lk == "asr" or lk.endswith("_asr") or "attack_success" in lk):
                yield next_path, float(value)
            else:
                yield from _walk_asr(value, next_path)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk_asr(value, f"{path}[{i}]")


def summarize_attack_risk(attack_metrics: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not attack_metrics:
        return {"available": False, "n_asr_values": 0, "max_asr": 0.0, "worst_case": None}
    values = [(p, v) for p, v in _walk_asr(attack_metrics) if 0.0 <= float(v) <= 1.0]
    if not values:
        return {"available": True, "n_asr_values": 0, "max_asr": 0.0, "worst_case": None}
    worst = max(values, key=lambda x: x[1])
    return {
        "available": True,
        "n_asr_values": len(values),
        "max_asr": float(worst[1]),
        "worst_case": {"path": worst[0], "asr": float(worst[1])},
        "top_asr": [{"path": p, "asr": float(v)} for p, v in sorted(values, key=lambda x: x[1], reverse=True)[:10]],
    }


def decide_acceptance(
    before_report: Dict[str, Any],
    after_report: Dict[str, Any],
    before_metrics: dict | None = None,
    after_metrics: dict | None = None,
    max_map_drop: float = 0.03,
    min_fp_reduction: float = 0.8,
    detox_manifest: dict | None = None,
    allow_weak_supervision: bool = False,
    attack_metrics: dict | None = None,
    max_allowed_asr: float | None = 0.20,
    safety_critical: bool = False,
    require_green: bool = False,
    require_clean_metrics: bool = False,
) -> Dict[str, Any]:
    """Make an explicit deployment acceptance decision after detox.

    In addition to risk score and mAP preservation, this gate can enforce attack
    regression metrics (ASR), weak-supervision boundaries, and stricter
    safety-critical deployment policy.
    """
    security_cmp = compare_security_reports(before_report, after_report)
    metric_cmp = compare_yolo_metrics(before_metrics, after_metrics) if before_metrics is not None and after_metrics is not None else {"available": False}
    supervision_cmp = summarize_supervision_risk(detox_manifest)
    attack_cmp = summarize_attack_risk(attack_metrics)
    warnings: list[str] = []

    after_level = security_cmp["risk_after"]
    after_rank = security_cmp["risk_rank_after"]
    if after_rank >= RISK_ORDER["Red"]:
        warnings.append(f"after risk is still {after_level}")
    if require_green or safety_critical:
        if after_rank != RISK_ORDER["Green"]:
            warnings.append(f"safety gate requires Green, got {after_level}")
    if not security_cmp["risk_not_worse"]:
        warnings.append("security risk worsened after detox")
    if supervision_cmp.get("weak_supervision") and not allow_weak_supervision:
        warnings.append(
            "weak supervision mode cannot be accepted as fully safe without explicit override: "
            + str(supervision_cmp.get("reason") or "unknown weak supervision")
        )
    if supervision_cmp.get("verification_status") == "failed":
        warnings.append("automatic verification failed in detox manifest")
    if safety_critical and detox_manifest and supervision_cmp.get("verification_status") != "completed":
        warnings.append("safety-critical gate requires completed automatic verification")

    fp_before = float(security_cmp.get("fp_proxy_before", 0.0) or 0.0)
    fp_reduction = float(security_cmp.get("fp_proxy_reduction", 0.0) or 0.0)
    if fp_before > 1e-9 and fp_reduction < float(min_fp_reduction):
        warnings.append(f"FP/backdoor proxy reduction {fp_reduction:.3f} is below required {min_fp_reduction:.3f}")

    map_drop = None
    if metric_cmp.get("available"):
        map_drop = metric_cmp.get("map_drop")
        if map_drop is not None and float(map_drop) > float(max_map_drop):
            warnings.append(f"mAP50-95 drop {float(map_drop):.4f} exceeds max_map_drop {max_map_drop:.4f}")
    elif require_clean_metrics or safety_critical:
        warnings.append("clean validation metrics are required for this acceptance policy")

    if attack_cmp.get("available") and max_allowed_asr is not None:
        max_asr = float(attack_cmp.get("max_asr", 0.0) or 0.0)
        if max_asr > float(max_allowed_asr):
            warnings.append(f"attack regression ASR {max_asr:.3f} exceeds max_allowed_asr {float(max_allowed_asr):.3f}")
    elif safety_critical:
        warnings.append("safety-critical gate requires attack regression / ASR metrics")

    accepted = len(warnings) == 0
    if accepted:
        if security_cmp["risk_improved"] and metric_cmp.get("available"):
            reason = "risk reduced and clean metric preserved"
        elif security_cmp["risk_improved"]:
            reason = "risk reduced; clean metrics unavailable"
        else:
            reason = "risk acceptable and not worsened"
    else:
        reason = "; ".join(warnings)

    accepted_for_quarantine_reduction = bool(
        security_cmp["risk_not_worse"]
        and after_rank <= RISK_ORDER["Yellow"]
        and not any("ASR" in w and "exceeds" in w for w in warnings)
    )
    if accepted:
        operational_status = "deployable"
    elif accepted_for_quarantine_reduction:
        operational_status = "risk_reduction_only"
    else:
        operational_status = "blocked"

    return {
        "accepted": accepted,
        "accepted_for_deployment": accepted,
        "accepted_for_quarantine_reduction": accepted_for_quarantine_reduction,
        "operational_status": operational_status,
        "reason": reason,
        "risk_before": security_cmp["risk_before"],
        "risk_after": security_cmp["risk_after"],
        "score_before": security_cmp.get("score_before"),
        "score_after": security_cmp.get("score_after"),
        "map_drop": map_drop,
        "map50_drop": metric_cmp.get("map50_drop") if metric_cmp.get("available") else None,
        "fp_proxy_before": security_cmp.get("fp_proxy_before"),
        "fp_proxy_after": security_cmp.get("fp_proxy_after"),
        "fp_proxy_reduction": security_cmp.get("fp_proxy_reduction"),
        "max_asr": attack_cmp.get("max_asr"),
        "warnings": warnings,
        "security_compare": security_cmp,
        "metric_compare": metric_cmp,
        "supervision_compare": supervision_cmp,
        "attack_compare": attack_cmp,
    }
