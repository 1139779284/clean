from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .stats import wilson_interval


def load_json(path_or_obj: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if path_or_obj is None:
        return {}
    if isinstance(path_or_obj, Mapping):
        return dict(path_or_obj)
    return json.loads(Path(path_or_obj).read_text(encoding="utf-8"))


def normalize_attack_name(raw: str) -> str:
    name = str(raw)
    if "::" in name:
        name = name.split("::")[-1]
    return name


def extract_asr_matrix(report: Mapping[str, Any] | None) -> dict[str, float]:
    """Extract attack -> ASR from common external hard-suite report shapes."""

    data = dict(report or {})
    out: dict[str, float] = {}
    matrix = data.get("asr_matrix") or (data.get("summary") or {}).get("asr_matrix") or {}
    if isinstance(matrix, Mapping):
        for k, v in matrix.items():
            try:
                out[normalize_attack_name(str(k))] = float(v)
            except (TypeError, ValueError):
                pass
    for item in data.get("top_attacks") or (data.get("summary") or {}).get("top_attacks") or []:
        if isinstance(item, Mapping) and "attack" in item and ("asr" in item or "attack_success_rate" in item):
            try:
                out[normalize_attack_name(str(item["attack"]))] = float(item.get("asr", item.get("attack_success_rate")))
            except (TypeError, ValueError):
                pass
    return out


def extract_counts(report: Mapping[str, Any] | None) -> dict[str, tuple[int, int]]:
    """Extract attack -> (successes, total) when reports include counts."""

    data = dict(report or {})
    counts: dict[str, tuple[int, int]] = {}
    # Preferred shape emitted by some audits.
    success_counts = data.get("success_counts") or data.get("attack_success_counts") or {}
    totals = data.get("counts") or data.get("attack_counts") or {}
    if isinstance(success_counts, Mapping) and isinstance(totals, Mapping):
        for k, s in success_counts.items():
            if k in totals:
                try:
                    counts[normalize_attack_name(str(k))] = (int(s), int(totals[k]))
                except (TypeError, ValueError):
                    pass
    # top_attacks rows often carry n and asr.
    for item in data.get("top_attacks") or (data.get("summary") or {}).get("top_attacks") or []:
        if not isinstance(item, Mapping):
            continue
        attack = item.get("attack")
        n = item.get("n") or item.get("total") or item.get("count")
        asr = item.get("asr") or item.get("attack_success_rate")
        if attack is None or n is None or asr is None:
            continue
        try:
            total = int(n)
            successes = int(round(float(asr) * total))
            counts[normalize_attack_name(str(attack))] = (successes, total)
        except (TypeError, ValueError):
            pass
    return counts


def summarize_external_report(report: Mapping[str, Any] | None, *, confidence: float = 0.95) -> dict[str, Any]:
    data = dict(report or {})
    matrix = extract_asr_matrix(data)
    counts = extract_counts(data)
    max_asr = data.get("max_asr", (data.get("summary") or {}).get("max_asr"))
    mean_asr = data.get("mean_asr", (data.get("summary") or {}).get("mean_asr"))
    if max_asr is None and matrix:
        max_asr = max(matrix.values())
    if mean_asr is None and matrix:
        mean_asr = sum(matrix.values()) / max(1, len(matrix))
    cis = {}
    for attack, (s, n) in counts.items():
        cis[attack] = wilson_interval(s, n, confidence).to_dict()
    return {
        "max_asr": float(max_asr or 0.0),
        "mean_asr": float(mean_asr or 0.0),
        "asr_matrix": matrix,
        "counts": {k: {"successes": v[0], "total": v[1]} for k, v in counts.items()},
        "wilson_ci": cis,
    }


def compare_guarded_unguarded(*, unguarded: Mapping[str, Any] | None, guarded: Mapping[str, Any] | None) -> dict[str, Any]:
    u = summarize_external_report(unguarded or {})
    g = summarize_external_report(guarded or {})
    attacks = sorted(set(u["asr_matrix"]) | set(g["asr_matrix"]))
    rows = []
    for attack in attacks:
        ua = float(u["asr_matrix"].get(attack, 0.0))
        ga = float(g["asr_matrix"].get(attack, 0.0))
        rows.append({"attack": attack, "unguarded_asr": ua, "guarded_asr": ga, "guard_delta": ga - ua, "guard_reduction": ua - ga})
    return {"unguarded": u, "guarded": g, "attacks": rows, "guard_is_primary": False}

# Compatibility alias for newer algorithm orchestration scripts.
def summarize_asr(report: Mapping[str, Any] | None) -> dict[str, Any]:
    s = summarize_external_report(report or {})
    return {"attack_asr": dict(s.get("asr_matrix", {})), "max_asr": float(s.get("max_asr", 0.0)), "mean_asr": float(s.get("mean_asr", 0.0)), "n_attacks": len(s.get("asr_matrix", {})), "guarded": bool(((report or {}).get("config") or {}).get("apply_overlap_class_guard") or ((report or {}).get("config") or {}).get("apply_semantic_abstain"))}
