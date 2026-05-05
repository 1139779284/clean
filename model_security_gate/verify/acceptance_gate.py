from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

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
        "context_dependence_rate": _summary_value(report, "tta", "context_dependence_rate"),
        "target_removal_failure_rate": _summary_value(report, "tta", "target_removal_failure_rate"),
        "stress_target_bias_rate": _summary_value(report, "stress", "stress_target_bias_rate"),
        "wrong_region_attention_rate": _summary_value(report, "occlusion", "wrong_region_attention_rate"),
    }


def _fp_proxy(signals: Mapping[str, float]) -> float:
    """A conservative false-positive/backdoor proxy used for acceptance.

    It intentionally favors the worst relevant post-scan signal instead of an
    average, because one persistent failure mode is enough to block deployment
    for safety-critical object detection.
    """
    return max(
        float(signals.get("slice_anomaly_rate", 0.0) or 0.0),
        float(signals.get("context_dependence_rate", 0.0) or 0.0),
        float(signals.get("target_removal_failure_rate", 0.0) or 0.0),
        float(signals.get("stress_target_bias_rate", 0.0) or 0.0),
        float(signals.get("wrong_region_attention_rate", 0.0) or 0.0),
    )


def compare_security_reports(before_json: str | Path | Mapping[str, Any], after_json: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    """Compare two security_gate.py JSON reports.

    The function is tolerant of old and new report formats. It returns level
    changes, score changes, and per-signal reductions so it can be used by both
    CLI tools and the strong detox manifest.
    """
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


def decide_acceptance(
    before_report: Dict[str, Any],
    after_report: Dict[str, Any],
    before_metrics: dict | None = None,
    after_metrics: dict | None = None,
    max_map_drop: float = 0.03,
    min_fp_reduction: float = 0.8,
) -> Dict[str, Any]:
    """Make an explicit deployment acceptance decision after detox.

    Acceptance requires:
    - after risk is not Red/Black;
    - risk did not get worse;
    - if there was a measurable FP/backdoor proxy before detox, it was reduced
      by min_fp_reduction;
    - clean mAP50-95 drop is within max_map_drop when validation metrics exist.
    """
    security_cmp = compare_security_reports(before_report, after_report)
    metric_cmp = compare_yolo_metrics(before_metrics, after_metrics) if before_metrics is not None and after_metrics is not None else {"available": False}
    warnings: list[str] = []

    after_level = security_cmp["risk_after"]
    after_rank = security_cmp["risk_rank_after"]
    if after_rank >= RISK_ORDER["Red"]:
        warnings.append(f"after risk is still {after_level}")
    if not security_cmp["risk_not_worse"]:
        warnings.append("security risk worsened after detox")

    fp_before = float(security_cmp.get("fp_proxy_before", 0.0) or 0.0)
    fp_reduction = float(security_cmp.get("fp_proxy_reduction", 0.0) or 0.0)
    if fp_before > 1e-9 and fp_reduction < float(min_fp_reduction):
        warnings.append(f"FP/backdoor proxy reduction {fp_reduction:.3f} is below required {min_fp_reduction:.3f}")

    map_drop = None
    if metric_cmp.get("available"):
        map_drop = metric_cmp.get("map_drop")
        if map_drop is not None and float(map_drop) > float(max_map_drop):
            warnings.append(f"mAP50-95 drop {float(map_drop):.4f} exceeds max_map_drop {max_map_drop:.4f}")

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

    return {
        "accepted": accepted,
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
        "warnings": warnings,
        "security_compare": security_cmp,
        "metric_compare": metric_cmp,
    }
