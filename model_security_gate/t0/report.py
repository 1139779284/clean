from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_gate import T0EvidenceGateConfig, evaluate_t0_evidence
from .green_profiles import build_green_profile_scorecard
from .matrix_aggregator import (
    MatrixAggregatorConfig,
    aggregate_matrix_entries,
    render_matrix_aggregate_markdown,
)
from .metrics import compare_guarded_unguarded, load_json
from .residuals import build_frontier_plan


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _load_matrix_entries(
    matrix_summaries: Sequence[str | Path | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Load and deduplicate poison-matrix summary entries.

    Accepts paths or dicts; each payload is expected to have
    ``{"entries": [{...}, ...]}`` like the summaries written by
    ``scripts/train_t0_poison_models_yolo.py``.  Duplicate entries (same run /
    same weights path) are kept only once so multiple batch summaries can be
    merged without double counting.
    """

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in matrix_summaries or ():
        if source is None:
            continue
        data: Mapping[str, Any] = (
            source if isinstance(source, Mapping) else json.loads(Path(source).read_text(encoding="utf-8"))
        )
        entries = data.get("entries") if isinstance(data, Mapping) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            entry_dict = dict(entry)
            key = str(
                entry_dict.get("run")
                or entry_dict.get("weights")
                or json.dumps(entry_dict, sort_keys=True)
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry_dict)
    return merged


def _render_matrix_aggregate_section(aggregate: Mapping[str, Any]) -> list[str]:
    overall = aggregate.get("overall") or {}
    strong = overall.get("strong_cell_pass_rate") or {}
    usable = overall.get("usable_cell_pass_rate") or {}
    lines: list[str] = []
    lines.append(f"- status: `{aggregate.get('status')}`")
    lines.append(f"- entries: `{aggregate.get('n_entries')}`")
    lines.append(
        "- strong cell pass: `{s}/{t} = {rate:.4f}` "
        "[`{low:.4f}`, `{high:.4f}`]".format(
            s=int(strong.get("successes", 0)),
            t=int(strong.get("total", 0)),
            rate=float(strong.get("rate", 0.0)),
            low=float(strong.get("low", 0.0)),
            high=float(strong.get("high", 0.0)),
        )
    )
    lines.append(
        "- usable cell pass: `{s}/{t} = {rate:.4f}` "
        "[`{low:.4f}`, `{high:.4f}`]".format(
            s=int(usable.get("successes", 0)),
            t=int(usable.get("total", 0)),
            rate=float(usable.get("rate", 0.0)),
            low=float(usable.get("low", 0.0)),
            high=float(usable.get("high", 0.0)),
        )
    )
    warnings = aggregate.get("warnings") or []
    if warnings:
        lines.append("- warnings:")
        lines.extend([f"  - {msg}" for msg in warnings])
    else:
        lines.append("- warnings: None")
    lines.append("")
    lines.append(
        "| attack | status | cells | seeds | max ASR | mean ASR | CV | strong pass | off-target delta |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for attack, row in (aggregate.get("per_attack") or {}).items():
        interval = row.get("strong_pass_rate") or {}
        bleed = row.get("bleed_over") or {}
        lines.append(
            "| `{attack}` | `{status}` | `{cells}` | `{seeds}` | "
            "`{max_asr:.4f}` | `{mean_asr:.4f}` | `{cv:.4f}` | "
            "`{srate:.4f}` (`{s}/{t}`) | `{delta:.4f}` |".format(
                attack=attack,
                status=row.get("status"),
                cells=row.get("n_cells", 0),
                seeds=row.get("n_seeds", 0),
                max_asr=float(row.get("max_intended_asr", 0.0)),
                mean_asr=float(row.get("mean_intended_asr", 0.0)),
                cv=float(row.get("cv_intended_asr", 0.0)),
                srate=float(interval.get("rate", 0.0)),
                s=int(interval.get("successes", 0)),
                t=int(interval.get("total", 0)),
                delta=float(bleed.get("max_offtarget_delta", 0.0)),
            )
        )
    return lines


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
    poison_matrix_summaries: Sequence[str | Path | Mapping[str, Any]] | None = None,
    matrix_config: MatrixAggregatorConfig | None = None,
    write_full_matrix_report: bool = True,
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

    matrix_entries = _load_matrix_entries(poison_matrix_summaries)
    matrix_aggregate: dict[str, Any] | None = None
    matrix_report_path: Path | None = None
    if matrix_entries:
        matrix_aggregate = aggregate_matrix_entries(matrix_entries, cfg=matrix_config)
        if write_full_matrix_report:
            matrix_report_path = out / "T0_POISON_MATRIX_AGGREGATE.md"
            matrix_report_path.write_text(
                render_matrix_aggregate_markdown(matrix_aggregate), encoding="utf-8"
            )

    payload: dict[str, Any] = {
        "gate": gate,
        "green_profiles": green_profiles,
        "guarded_vs_unguarded": comparison,
        "frontier_plan": plan,
    }
    if matrix_aggregate is not None:
        payload["matrix_aggregate"] = matrix_aggregate
    (out / "t0_evidence_pack.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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

    if matrix_aggregate is not None:
        lines.append(_section("Poison-Matrix Aggregate Evidence"))
        lines.extend(_render_matrix_aggregate_section(matrix_aggregate))
        if matrix_report_path is not None:
            lines.append("")
            lines.append(f"Full aggregate report: `{matrix_report_path.name}`")

    lines.append("\n")
    (out / "T0_EVIDENCE_PACK.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
