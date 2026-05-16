"""T0 poison-matrix aggregate evidence.

This module consumes a poison-model matrix summary (the JSON written by
``scripts/train_t0_poison_models_yolo.py`` / the existing matrix summaries in
``runs/t0_poison_model_matrix_summary_*``) and produces publication-grade
aggregate statistics:

* per-attack Wilson-95 pass rates for strong/usable poison models;
* per-attack mean/median/std/CV intended ASR;
* per-attack dose-response curves (mean ASR vs poison rate, with monotonicity
  check and per-rate seed CV);
* off-target bleed-over matrices (a poison model targeting attack A should not
  simultaneously succeed on attacks B, C, ...);
* explicit warnings when dose-response is non-monotonic, no strong cell exists,
  or a non-target attack exceeds the configured off-target bound.

The module is intentionally lightweight: no torch, no ultralytics, pure Python
+ project statistics helpers, so it runs inside ``pixi run ci-smoke``.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stats import wilson_interval, zero_failure_upper_bound


# ---------------------------------------------------------------------------
# Attack-name helpers mirror the conventions in poison_matrix_evidence.py so
# the two modules stay consistent.  We duplicate them (tiny, stable code)
# instead of importing private names across modules.
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return str(name).replace("\\", "/").split("::")[-1].strip().lower()


def _attack_aliases(attack: str) -> tuple[str, ...]:
    name = _norm(attack)
    aliases = {name}
    if name == "semantic_green_cleanlabel":
        aliases.add("semantic_cleanlabel")
    if name == "semantic_cleanlabel":
        aliases.add("semantic_green_cleanlabel")
    if name.startswith("badnet_oga"):
        aliases.add("badnet_oga")
    if name.startswith("wanet_oga"):
        aliases.add("wanet_oga")
    return tuple(sorted(aliases))


def _intended_asr(matrix: Mapping[str, float], attack: str) -> float:
    normalized = {_norm(k): float(v) for k, v in matrix.items()}
    for alias in _attack_aliases(attack):
        if alias in normalized:
            return normalized[alias]
    key = _norm(attack)
    for k, v in normalized.items():
        if k.startswith(key) or key.startswith(k):
            return float(v)
    return 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixAggregatorConfig:
    """Aggregation thresholds.

    Defaults match the research gate used elsewhere in the project:

    * ``strong_asr_threshold = 0.20`` — a poison cell must reach at least this
      intended ASR to be counted as a strong poison model;
    * ``usable_asr_threshold = 0.05`` — matches ``min_usable_asr`` in
      ``PoisonMatrixEvidenceConfig``;
    * ``off_target_warn_absolute = 0.20`` — a non-target attack exceeding this
      value emits a bleed-over warning;
    * ``dose_response_tolerance`` allows small per-rate noise without
      triggering a false non-monotonic warning.
    """

    strong_asr_threshold: float = 0.20
    usable_asr_threshold: float = 0.05
    confidence: float = 0.95
    off_target_warn_absolute: float = 0.20
    off_target_warn_delta: float = 0.15
    min_seeds_for_stability: int = 2
    dose_response_tolerance: float = 0.02


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _status_for_asr(asr: float, cfg: MatrixAggregatorConfig) -> str:
    if asr >= cfg.strong_asr_threshold:
        return "strong"
    if asr >= cfg.usable_asr_threshold:
        return "weak"
    return "blocked"


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _cv(values: Sequence[float]) -> float:
    m = _mean(values)
    if m == 0.0:
        return 0.0
    return _std(values) / m


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _load_entries(summary: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    data: Mapping[str, Any]
    if isinstance(summary, (str, Path)):
        data = json.loads(Path(summary).read_text(encoding="utf-8"))
    else:
        data = summary
    entries = data.get("entries") if isinstance(data, Mapping) else None
    if not isinstance(entries, list):
        return []
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def _extract_intended_and_matrix(
    entry: Mapping[str, Any],
    attack: str,
) -> tuple[float, dict[str, float]]:
    """Return (intended_asr, normalized asr matrix) for an entry.

    The summary JSON produced by ``train_t0_poison_models_yolo.py`` stores
    per-attack ASR keyed as ``"suite_name::attack_name"``.  We flatten that to
    just the attack name so bleed-over comparisons work across suites.
    """

    flat: dict[str, float] = {}
    raw = entry.get("asr_matrix") or {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                flat[_norm(str(key))] = float(value)
            except (TypeError, ValueError):
                continue
    if not flat:
        # Fall back to ``top_attacks`` rows.
        for item in entry.get("top_attacks") or []:
            if not isinstance(item, Mapping):
                continue
            name = item.get("attack")
            asr = item.get("asr")
            if name is None or asr is None:
                continue
            try:
                flat[_norm(str(name))] = float(asr)
            except (TypeError, ValueError):
                continue
    intended = _intended_asr(flat, attack)
    return intended, flat


def _per_cell_rows(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        attack = str(
            entry.get("attack")
            or entry.get("attack_name")
            or entry.get("run")
            or "unknown"
        )
        intended, matrix = _extract_intended_and_matrix(entry, attack)
        poison_rate = entry.get("poison_rate")
        seed = entry.get("seed")
        try:
            poison_rate_f = float(poison_rate) if poison_rate is not None else None
        except (TypeError, ValueError):
            poison_rate_f = None
        try:
            seed_i = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed_i = None
        rows.append(
            {
                "attack": attack,
                "run": str(entry.get("run") or entry.get("name") or attack),
                "tier": entry.get("tier"),
                "poison_rate": poison_rate_f,
                "seed": seed_i,
                "epochs": entry.get("epochs"),
                "weights": entry.get("weights"),
                "report": entry.get("report"),
                "weights_exists": bool(entry.get("weights_exists", False)),
                "report_exists": bool(entry.get("report_exists", False)),
                "intended_asr": float(intended),
                "max_asr": float(entry.get("max_asr") or intended),
                "mean_asr": float(entry.get("mean_asr") or 0.0),
                "asr_matrix": matrix,
            }
        )
    return rows


def _pass_rate_row(values: Sequence[bool], *, confidence: float) -> dict[str, Any]:
    total = len(values)
    successes = int(sum(1 for x in values if bool(x)))
    interval = wilson_interval(successes, total, confidence).to_dict()
    if successes == 0:
        interval["zero_failure_upper_bound"] = zero_failure_upper_bound(total, confidence) if total > 0 else 1.0
    return interval


def _seed_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_rate: dict[float, list[float]] = {}
    for row in rows:
        rate = row.get("poison_rate")
        if rate is None:
            continue
        by_rate.setdefault(float(rate), []).append(float(row.get("intended_asr", 0.0)))
    summary: dict[str, Any] = {}
    for rate, values in sorted(by_rate.items()):
        summary[f"{rate:g}"] = {
            "n_seeds": len(values),
            "mean_asr": _mean(values),
            "std_asr": _std(values),
            "cv_asr": _cv(values),
            "min_asr": float(min(values)) if values else 0.0,
            "max_asr": float(max(values)) if values else 0.0,
        }
    return summary


def _dose_response(rows: Sequence[Mapping[str, Any]], tolerance: float) -> dict[str, Any]:
    by_rate: dict[float, list[float]] = {}
    for row in rows:
        rate = row.get("poison_rate")
        if rate is None:
            continue
        by_rate.setdefault(float(rate), []).append(float(row.get("intended_asr", 0.0)))
    curve = []
    for rate in sorted(by_rate):
        values = by_rate[rate]
        curve.append(
            {
                "poison_rate": rate,
                "n": len(values),
                "mean_asr": _mean(values),
                "max_asr": float(max(values)) if values else 0.0,
            }
        )
    # Monotonicity: consecutive mean_asr should be non-decreasing within tolerance.
    non_monotonic: list[dict[str, Any]] = []
    for prev, cur in zip(curve, curve[1:]):
        drop = prev["mean_asr"] - cur["mean_asr"]
        if drop > float(tolerance):
            non_monotonic.append(
                {
                    "from_rate": prev["poison_rate"],
                    "to_rate": cur["poison_rate"],
                    "mean_asr_drop": drop,
                }
            )
    return {
        "curve": curve,
        "is_monotonic": not non_monotonic,
        "violations": non_monotonic,
        "n_rates": len(curve),
    }


def _bleed_over_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    intended_attack: str,
) -> dict[str, Any]:
    """Average off-target ASR across every cell that targets ``intended_attack``.

    Each ``rows`` entry contributes its full ASR matrix.  We report:

    * ``per_attack``: mean off-target ASR per (non-intended) attack;
    * ``max_offtarget_attack`` / ``max_offtarget_asr``;
    * ``max_offtarget_delta``: mean intended ASR minus max off-target mean ASR.
    """

    if not rows:
        return {"per_attack": {}, "n_cells": 0, "max_offtarget_attack": None, "max_offtarget_asr": 0.0, "max_offtarget_delta": 0.0}
    intended_aliases = set(_attack_aliases(intended_attack))
    attack_values: dict[str, list[float]] = {}
    for row in rows:
        matrix = row.get("asr_matrix") or {}
        for name, value in matrix.items():
            if _norm(name) in intended_aliases:
                continue
            attack_values.setdefault(_norm(name), []).append(float(value))
    per_attack = {name: _mean(vals) for name, vals in attack_values.items()}
    mean_intended = _mean([float(row.get("intended_asr", 0.0)) for row in rows])
    max_attack = max(per_attack, key=per_attack.get) if per_attack else None
    max_value = float(per_attack[max_attack]) if max_attack else 0.0
    return {
        "per_attack": per_attack,
        "n_cells": len(rows),
        "mean_intended_asr": mean_intended,
        "max_offtarget_attack": max_attack,
        "max_offtarget_asr": max_value,
        "max_offtarget_delta": mean_intended - max_value,
    }


def aggregate_matrix_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    cfg: MatrixAggregatorConfig | None = None,
) -> dict[str, Any]:
    """Aggregate a list of poison-matrix entries into per-attack statistics.

    Inputs follow the shape of ``t0_poison_model_matrix_summary.json``:

    ``{"entries": [{"attack": ..., "poison_rate": ..., "seed": ...,
                    "asr_matrix": {...}, "max_asr": ..., "mean_asr": ...}, ...]}``

    Returns a nested dict with per-attack pass rates, dose-response curves,
    seed stability, and bleed-over.  Missing fields degrade gracefully.
    """

    cfg = cfg or MatrixAggregatorConfig()
    rows = _per_cell_rows(entries)

    attacks = sorted({row["attack"] for row in rows})
    per_attack: dict[str, Any] = {}
    warnings: list[str] = []

    for attack in attacks:
        attack_rows = [row for row in rows if row["attack"] == attack]
        intended_values = [float(row["intended_asr"]) for row in attack_rows]
        max_intended = float(max(intended_values)) if intended_values else 0.0
        mean_intended = _mean(intended_values)
        median_intended = float(statistics.median(intended_values)) if intended_values else 0.0
        strong_flags = [value >= cfg.strong_asr_threshold for value in intended_values]
        usable_flags = [value >= cfg.usable_asr_threshold for value in intended_values]
        strong_pass = _pass_rate_row(strong_flags, confidence=cfg.confidence)
        usable_pass = _pass_rate_row(usable_flags, confidence=cfg.confidence)
        dose = _dose_response(attack_rows, tolerance=cfg.dose_response_tolerance)
        seed_stability = _seed_stability(attack_rows)
        bleed = _bleed_over_row(attack_rows, intended_attack=attack)

        status = "blocked"
        if strong_pass["successes"] > 0:
            status = "strong"
        elif usable_pass["successes"] > 0:
            status = "usable"

        if status == "blocked":
            warnings.append(f"{attack}: no usable poison cell (max intended ASR {max_intended:.6g})")
        if strong_pass["successes"] > 0 and not dose["is_monotonic"]:
            violations = dose["violations"][0]
            warnings.append(
                f"{attack}: dose-response non-monotonic (rate {violations['from_rate']}"
                f" -> {violations['to_rate']}, mean drop {violations['mean_asr_drop']:.6g})"
            )
        if (
            bleed["max_offtarget_asr"] >= cfg.off_target_warn_absolute
            and bleed["max_offtarget_delta"] <= cfg.off_target_warn_delta
        ):
            warnings.append(
                f"{attack}: off-target bleed-over {bleed['max_offtarget_attack']}="
                f"{bleed['max_offtarget_asr']:.6g} close to intended mean {bleed['mean_intended_asr']:.6g}"
            )

        best_row = max(attack_rows, key=lambda r: r["intended_asr"], default=None)
        per_attack[attack] = {
            "status": status,
            "n_cells": len(attack_rows),
            "n_seeds": len({row["seed"] for row in attack_rows if row["seed"] is not None}),
            "n_poison_rates": len({row["poison_rate"] for row in attack_rows if row["poison_rate"] is not None}),
            "max_intended_asr": max_intended,
            "mean_intended_asr": mean_intended,
            "median_intended_asr": median_intended,
            "std_intended_asr": _std(intended_values),
            "cv_intended_asr": _cv(intended_values),
            "strong_pass_rate": strong_pass,
            "usable_pass_rate": usable_pass,
            "dose_response": dose,
            "seed_stability": seed_stability,
            "bleed_over": bleed,
            "best_run": best_row["run"] if best_row else None,
            "best_weights": best_row["weights"] if best_row else None,
        }

    total_cells = len(rows)
    strong_total = int(sum(1 for row in rows if row["intended_asr"] >= cfg.strong_asr_threshold))
    usable_total = int(sum(1 for row in rows if row["intended_asr"] >= cfg.usable_asr_threshold))
    overall = {
        "n_cells": total_cells,
        "n_attacks": len(attacks),
        "strong_cell_pass_rate": _pass_rate_row(
            [row["intended_asr"] >= cfg.strong_asr_threshold for row in rows],
            confidence=cfg.confidence,
        ),
        "usable_cell_pass_rate": _pass_rate_row(
            [row["intended_asr"] >= cfg.usable_asr_threshold for row in rows],
            confidence=cfg.confidence,
        ),
        "strong_cells": strong_total,
        "usable_cells": usable_total,
    }

    return {
        "status": "warned" if warnings else "passed",
        "n_entries": len(rows),
        "warnings": warnings,
        "per_attack": per_attack,
        "overall": overall,
        "config": asdict(cfg),
        "rows": rows,
    }


def aggregate_matrix_summary(
    summary: Mapping[str, Any] | str | Path,
    *,
    cfg: MatrixAggregatorConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: accept either a dict or a path to summary JSON."""

    return aggregate_matrix_entries(_load_entries(summary), cfg=cfg)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_ci(interval: Mapping[str, Any]) -> str:
    return (
        f"{interval.get('successes', 0)}/{interval.get('total', 0)}"
        f" = {_fmt_pct(float(interval.get('rate', 0.0)))}"
        f" [{_fmt_pct(float(interval.get('low', 0.0)))},"
        f" {_fmt_pct(float(interval.get('high', 0.0)))}]"
    )


def render_matrix_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    cfg = aggregate.get("config") or {}
    lines: list[str] = ["# T0 Poison Matrix Aggregate Evidence", ""]
    lines.append(f"- status: `{aggregate.get('status')}`")
    lines.append(f"- entries: `{aggregate.get('n_entries')}`")
    overall = aggregate.get("overall") or {}
    lines.append(f"- strong cell pass: {_fmt_ci(overall.get('strong_cell_pass_rate') or {})}")
    lines.append(f"- usable cell pass: {_fmt_ci(overall.get('usable_cell_pass_rate') or {})}")
    lines.append(
        f"- thresholds: strong>=`{cfg.get('strong_asr_threshold')}`, "
        f"usable>=`{cfg.get('usable_asr_threshold')}`, "
        f"confidence=`{cfg.get('confidence')}`"
    )
    warnings = aggregate.get("warnings") or []
    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.extend([f"- {msg}" for msg in warnings])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Per-Attack Summary")
    lines.append("")
    lines.append(
        "| attack | status | cells | seeds | rates | max ASR | mean ASR | CV | strong pass (95%) | dose monotonic |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for attack, row in (aggregate.get("per_attack") or {}).items():
        lines.append(
            "| `{attack}` | `{status}` | `{n_cells}` | `{n_seeds}` | `{n_rates}` | "
            "`{max_asr:.4f}` | `{mean_asr:.4f}` | `{cv:.4f}` | {strong} | `{mono}` |".format(
                attack=attack,
                status=row.get("status"),
                n_cells=row.get("n_cells", 0),
                n_seeds=row.get("n_seeds", 0),
                n_rates=row.get("n_poison_rates", 0),
                max_asr=float(row.get("max_intended_asr", 0.0)),
                mean_asr=float(row.get("mean_intended_asr", 0.0)),
                cv=float(row.get("cv_intended_asr", 0.0)),
                strong=_fmt_ci(row.get("strong_pass_rate") or {}),
                mono=bool((row.get("dose_response") or {}).get("is_monotonic", True)),
            )
        )

    lines.append("")
    lines.append("## Dose-Response Curves")
    for attack, row in (aggregate.get("per_attack") or {}).items():
        curve = (row.get("dose_response") or {}).get("curve") or []
        if not curve:
            continue
        lines.append("")
        lines.append(f"### `{attack}`")
        lines.append("")
        lines.append("| poison rate | n cells | mean ASR | max ASR |")
        lines.append("|---:|---:|---:|---:|")
        for point in curve:
            lines.append(
                "| `{rate}` | `{n}` | `{mean:.4f}` | `{mx:.4f}` |".format(
                    rate=point.get("poison_rate"),
                    n=point.get("n", 0),
                    mean=float(point.get("mean_asr", 0.0)),
                    mx=float(point.get("max_asr", 0.0)),
                )
            )

    lines.append("")
    lines.append("## Off-Target Bleed-Over")
    lines.append("")
    lines.append(
        "Each row reports the mean ASR of non-intended attacks for poison models "
        "that target the named attack.  A strong intended ASR paired with a comparably "
        "strong off-target ASR would imply the cell is not attack-specific."
    )
    lines.append("")
    lines.append("| intended attack | mean intended | worst off-target | worst off-target ASR | delta |")
    lines.append("|---|---:|---|---:|---:|")
    for attack, row in (aggregate.get("per_attack") or {}).items():
        bleed = row.get("bleed_over") or {}
        lines.append(
            "| `{attack}` | `{intended:.4f}` | `{name}` | `{asr:.4f}` | `{delta:.4f}` |".format(
                attack=attack,
                intended=float(bleed.get("mean_intended_asr", 0.0)),
                name=bleed.get("max_offtarget_attack") or "-",
                asr=float(bleed.get("max_offtarget_asr", 0.0)),
                delta=float(bleed.get("max_offtarget_delta", 0.0)),
            )
        )

    return "\n".join(lines) + "\n"


def write_matrix_aggregate(out_dir: str | Path, aggregate: Mapping[str, Any]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "t0_poison_matrix_aggregate.json"
    md_path = out / "T0_POISON_MATRIX_AGGREGATE.md"
    json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_matrix_aggregate_markdown(aggregate), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "MatrixAggregatorConfig",
    "aggregate_matrix_entries",
    "aggregate_matrix_summary",
    "render_matrix_aggregate_markdown",
    "write_matrix_aggregate",
]
