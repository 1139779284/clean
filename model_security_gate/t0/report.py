from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_gate import T0EvidenceGateConfig, evaluate_t0_evidence
from .green_profiles import build_green_profile_scorecard
from .metrics import compare_guarded_unguarded, load_json
from .residuals import build_frontier_plan


def _section(title: str) -> str:
    return f"\n## {title}\n"


def build_t0_evidence_pack(
    *,
    out_dir: str | Path,
    guard_free_external: str | Path | Mapping[str, Any] | None = None,
    guarded_external: str | Path | Mapping[str, Any] | None = None,
    trigger_only_external: str | Path | Mapping[str, Any] | None = None,
    clean_metrics_before: str | Path | Mapping[str, Any] | None = None,
    clean_metrics_after: str | Path | Mapping[str, Any] | None = None,
    benchmark_audit: str | Path | Mapping[str, Any] | None = None,
    heldout_leakage: str | Path | Mapping[str, Any] | None = None,
    cfg: T0EvidenceGateConfig | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gf = load_json(guard_free_external)
    gd = load_json(guarded_external)
    to = load_json(trigger_only_external)
    cmb = load_json(clean_metrics_before)
    cma = load_json(clean_metrics_after)
    ba = load_json(benchmark_audit)
    hl = load_json(heldout_leakage)
    gate = evaluate_t0_evidence(
        guard_free_external=gf or None,
        guarded_external=gd or None,
        trigger_only_external=to or None,
        clean_metrics_before=cmb or None,
        clean_metrics_after=cma or None,
        benchmark_audit=ba or None,
        heldout_leakage=hl or None,
        cfg=cfg,
    )
    comparison = compare_guarded_unguarded(unguarded=gf or None, guarded=gd or None) if (gf or gd) else {}
    plan = build_frontier_plan([r for r in [gf, gd, to] if r], min_asr=0.001)
    green_profiles = build_green_profile_scorecard(gate, comparison)
    payload = {
        "gate": gate,
        "green_profiles": green_profiles,
        "guarded_vs_unguarded": comparison,
        "frontier_plan": plan,
    }
    (out / "t0_evidence_pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# T0 Evidence Pack\n")
    lines.append(f"Tier: `{gate.get('tier')}`  ")
    lines.append(f"Accepted for T0-style claim: `{gate.get('accepted')}`\n")
    lines.append(_section("Blocked Reasons"))
    if gate.get("blocked_reasons"):
        lines.extend([f"- {x}" for x in gate["blocked_reasons"]])
    else:
        lines.append("- None")
    lines.append(_section("Warnings"))
    if gate.get("warnings"):
        lines.extend([f"- {x}" for x in gate["warnings"]])
    else:
        lines.append("- None")
    lines.append(_section("Key Metrics"))
    metrics = gate.get("metrics", {})
    for name in ["guard_free", "guarded", "trigger_only"]:
        item = metrics.get(name) or {}
        if item:
            lines.append(f"- {name}: max_asr={item.get('max_asr')}, mean_asr={item.get('mean_asr')}")
    lines.append(f"- mAP50-95 drop: {metrics.get('map50_95_drop')}")
    lines.append(_section("Green Claim Profiles"))
    for row in green_profiles.get("profiles", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark} `{row.get('name')}`: {row.get('claim_type')} ({row.get('evidence_key')})")
    split = green_profiles.get("contribution_split", {})
    lines.append(_section("Guarded Safety vs Model Detox Contribution"))
    lines.append(f"- model_detox_primary: `{split.get('model_detox_primary')}`")
    lines.append(f"- guard_is_primary: `{split.get('guard_is_primary')}`")
    lines.append(f"- guard_max_asr_reduction: `{split.get('guard_max_asr_reduction')}`")
    lines.append(_section("Recommended Frontier Phase Order"))
    for phase in plan.get("recommended_phase_order", []):
        lines.append(f"- {phase}")
    lines.append(_section("Top Residuals"))
    for row in plan.get("top_residuals", []):
        lines.append(f"- {row['attack']}: ASR={row['asr']:.6g}, phase={row['phase']}")
    lines.append("\n")
    (out / "T0_EVIDENCE_PACK.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
