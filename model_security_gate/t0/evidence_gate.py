from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .metrics import summarize_external_report
from .stats import required_zero_failure_n, wilson_interval, zero_failure_upper_bound


@dataclass
class T0EvidenceGateConfig:
    """Strict evidence gate used for T0-style claims.

    T0 does not mean a guard-assisted demo.  The primary claim must be based on
    guard-free, corrected, leakage-free benchmark evidence, with guarded metrics
    reported only as a deployment layer.
    """

    max_guard_free_asr: float = 0.05
    max_guard_free_mean_asr: float = 0.02
    max_guarded_asr: float = 0.05
    max_trigger_only_asr: float = 0.05
    max_clean_map50_95_drop: float = 0.03
    require_integrity_pass: bool = True
    require_no_heldout_leakage: bool = True
    require_guard_free_primary: bool = True
    confidence: float = 0.95
    max_wilson_upper_for_t0: float = 0.05
    zero_failure_required_rate: float = 0.01
    required_attack_asr: dict[str, float] = field(default_factory=lambda: {
        "badnet_oda": 0.05,
        "badnet_oga": 0.05,
        "blend_oga": 0.05,
        "semantic_green_cleanlabel": 0.05,
        "wanet_oga": 0.05,
    })


def _metric(obj: Mapping[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    data = obj or {}
    for key in keys:
        cur: Any = data
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            try:
                return float(cur)
            except (TypeError, ValueError):
                return default
    return default


def _tier(blocked: Sequence[str], warnings: Sequence[str], *, has_guard_free: bool, ci_ready: bool) -> str:
    if blocked:
        return "not_t0_ready"
    if has_guard_free and ci_ready and not warnings:
        return "t0_candidate"
    if has_guard_free:
        return "t1_strong_candidate"
    return "engineering_green_only"


def evaluate_t0_evidence(
    *,
    guard_free_external: Mapping[str, Any] | None = None,
    guarded_external: Mapping[str, Any] | None = None,
    trigger_only_external: Mapping[str, Any] | None = None,
    clean_metrics_before: Mapping[str, Any] | None = None,
    clean_metrics_after: Mapping[str, Any] | None = None,
    benchmark_audit: Mapping[str, Any] | None = None,
    heldout_leakage: Mapping[str, Any] | None = None,
    cfg: T0EvidenceGateConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or T0EvidenceGateConfig()
    blocked: list[str] = []
    warnings: list[str] = []

    guard_free = summarize_external_report(guard_free_external or {}) if guard_free_external else None
    guarded = summarize_external_report(guarded_external or {}) if guarded_external else None
    trigger_only = summarize_external_report(trigger_only_external or {}) if trigger_only_external else None

    if cfg.require_integrity_pass:
        if not benchmark_audit:
            blocked.append("missing benchmark integrity audit")
        elif not bool(benchmark_audit.get("passed", False)):
            blocked.append("benchmark integrity audit failed")

    if cfg.require_no_heldout_leakage:
        leakage_count = int((heldout_leakage or {}).get("n_overlaps", (heldout_leakage or {}).get("detected_overlaps", 0)) or 0)
        if leakage_count > 0:
            blocked.append(f"held-out leakage detected: {leakage_count}")

    if cfg.require_guard_free_primary and not guard_free:
        blocked.append("missing guard-free corrected external result")
    if guard_free:
        if guard_free["max_asr"] > cfg.max_guard_free_asr:
            blocked.append(f"guard-free max ASR {guard_free['max_asr']:.6g} > {cfg.max_guard_free_asr:.6g}")
        if guard_free["mean_asr"] > cfg.max_guard_free_mean_asr:
            blocked.append(f"guard-free mean ASR {guard_free['mean_asr']:.6g} > {cfg.max_guard_free_mean_asr:.6g}")
        for attack, limit in cfg.required_attack_asr.items():
            if attack in guard_free["asr_matrix"] and guard_free["asr_matrix"][attack] > float(limit):
                blocked.append(f"guard-free {attack} ASR {guard_free['asr_matrix'][attack]:.6g} > {float(limit):.6g}")
    elif guarded:
        warnings.append("only guarded external result provided; this supports deployment safety, not weight-level purification")

    if guarded and guarded["max_asr"] > cfg.max_guarded_asr:
        warnings.append(f"guarded max ASR {guarded['max_asr']:.6g} > {cfg.max_guarded_asr:.6g}")
    if trigger_only and trigger_only["max_asr"] > cfg.max_trigger_only_asr:
        warnings.append(f"trigger-only max ASR {trigger_only['max_asr']:.6g} > {cfg.max_trigger_only_asr:.6g}")

    before_map = _metric(clean_metrics_before, "map50_95", "map", "metrics.map50_95")
    after_map = _metric(clean_metrics_after, "map50_95", "map", "metrics.map50_95")
    map_drop = None
    if before_map is not None and after_map is not None:
        map_drop = before_map - after_map
        if map_drop > cfg.max_clean_map50_95_drop:
            blocked.append(f"clean mAP50-95 drop {map_drop:.6g} > {cfg.max_clean_map50_95_drop:.6g}")
    else:
        warnings.append("clean mAP before/after metrics missing")

    ci_ready = False
    ci_summary: dict[str, Any] = {}
    if guard_free and guard_free.get("counts"):
        ci_ready = True
        for attack, pair in guard_free["counts"].items():
            s = int(pair["successes"])
            n = int(pair["total"])
            interval = wilson_interval(s, n, cfg.confidence).to_dict()
            interval["zero_failure_upper_bound_if_zero"] = zero_failure_upper_bound(n, cfg.confidence)
            if interval["high"] > cfg.max_wilson_upper_for_t0:
                ci_ready = False
                warnings.append(f"{attack} Wilson upper {interval['high']:.6g} > {cfg.max_wilson_upper_for_t0:.6g}")
            ci_summary[attack] = interval
    else:
        warnings.append("guard-free per-attack counts missing; cannot make confidence-bound T0 claim")

    tier = _tier(blocked, warnings, has_guard_free=bool(guard_free), ci_ready=ci_ready)
    return {
        "accepted": not blocked,
        "tier": tier,
        "blocked_reasons": blocked,
        "warnings": warnings,
        "metrics": {
            "guard_free": guard_free,
            "guarded": guarded,
            "trigger_only": trigger_only,
            "map50_95_before": before_map,
            "map50_95_after": after_map,
            "map50_95_drop": map_drop,
            "confidence_intervals": ci_summary,
            "zero_failure_samples_needed_for_rate": required_zero_failure_n(cfg.zero_failure_required_rate, cfg.confidence),
        },
        "config": asdict(cfg),
    }
