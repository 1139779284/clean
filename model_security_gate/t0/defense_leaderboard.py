"""T0 object-detection defense leaderboard.

This module implements a publication-grade, OD-specific defense comparison
layer that is stricter than BackdoorBench's single-number Defense
Effectiveness Rating (DER) reported for image classification:

* We adapt Backdoor\u00adBench-style (C-Acc, ASR, R-Acc, DER) [Wu et al., 2022
  https://arxiv.org/abs/2206.12654] to object detection.  Clean accuracy is
  replaced by clean mAP50-95.  ASR is read from the external hard-suite report
  (the project's benchmark is corrected, held-out-leak-audited, and
  goal-aware: OGA/ODA/RMA/semantic all supported).
* ASR is not reported as a single scalar: every attack gets a Wilson 95%
  confidence interval and a zero-failure upper bound (Clopper-Pearson style,
  already implemented in :mod:`.stats`).
* Each defended model is compared to its poisoned baseline with a paired
  McNemar test, per attack and aggregated.  This converts "defense A reduced
  ASR from 0.96 to 0.04" from a headline number into a hypothesis test.
* The composite score is ``OD-DER``: a strict-dominance variant that never
  accepts "improved ASR" alone - it also requires no per-attack regression,
  the clean mAP drop to respect a tolerance, and (when paired data exists) a
  significant paired ASR reduction.  This extends classification DER which
  only averages clean-acc drop and ASR drop.

The module is lightweight by design: no torch, no ultralytics, no SciPy.  It
runs under ``pixi run ci-smoke``.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import extract_asr_matrix, extract_counts, load_json, summarize_external_report
from .stats import wilson_interval, zero_failure_upper_bound


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefenseLeaderboardConfig:
    """Thresholds used for OD-DER scoring and acceptance.

    ``max_clean_map_drop`` follows the project's existing acceptance standard
    (3 percentage points).  ``max_per_attack_regression`` prevents a defense
    that improves average ASR while silently making one attack family worse
    from being ranked on top - BackdoorBench's per-attack table shows this is
    the typical failure mode for neural-cleanse / ANP baselines.
    ``min_paired_sig_alpha`` is the McNemar p-value required to call a
    per-attack ASR reduction statistically significant.
    """

    max_clean_map_drop: float = 0.03
    max_per_attack_regression: float = 0.00
    min_paired_sig_alpha: float = 0.05
    confidence: float = 0.95
    map_field: str = "map50_95"
    alt_map_fields: tuple[str, ...] = ("map", "metrics.map50_95", "map5095", "mAP50_95")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for McNemar's test.

    Given a 2x2 contingency table on matched pairs, ``b`` is the number of
    pairs where the baseline succeeded and the defended model did not, and
    ``c`` is the number of pairs where only the defended model succeeded.
    Under H0 (defense has no effect on this subset) the discordant count
    ``b + c`` splits Binomial(n, 0.5).  This routine returns the exact
    two-sided p-value without a normal approximation, so it is safe for the
    small per-attack suite sizes this project uses (typically 250-300 images).

    See Agresti, *Categorical Data Analysis* (2013), McNemar's test.
    """

    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    # Sum probabilities of outcomes at least as extreme.
    cumulative = 0.0
    for i in range(0, k + 1):
        cumulative += math.comb(n, i)
    p = cumulative / (2 ** n) * 2.0
    return min(1.0, max(0.0, p))


def _as_map(metrics: Mapping[str, Any] | None, cfg: DefenseLeaderboardConfig) -> float | None:
    if not metrics:
        return None
    for key in (cfg.map_field, *cfg.alt_map_fields):
        cur: Any = metrics
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
                continue
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DefenseEntry:
    """Single row in the leaderboard.

    ``poisoned_external``, ``defended_external`` and ``clean_*`` accept the
    exact JSON shape the rest of the project emits (external_hard_suite_asr
    reports and clean evaluation metrics).  They can be provided as dicts or
    as paths/strings resolved via :func:`metrics.load_json`.
    """

    name: str
    poisoned_model_id: str
    defense: str
    poisoned_external: Mapping[str, Any] | str | Path | None = None
    defended_external: Mapping[str, Any] | str | Path | None = None
    clean_before: Mapping[str, Any] | str | Path | None = None
    clean_after: Mapping[str, Any] | str | Path | None = None
    # Optional paired rows.  When both reports contain per-image rows keyed by
    # (attack, image_basename) we can run McNemar for every attack.
    use_paired_rows: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Strip heavy payloads when the user passed dicts; paths are always
        # serializable as-is.
        for k in ("poisoned_external", "defended_external", "clean_before", "clean_after"):
            v = d.get(k)
            if isinstance(v, (str, Path)):
                d[k] = str(v)
            else:
                d[k] = None if v is None else "<inline>"
        return d


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _rows_by_attack(report: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Group external_hard_suite rows by attack name then image basename."""

    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not report:
        return out
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        attack = str(row.get("attack") or "").strip()
        basename = str(row.get("image_basename") or row.get("image") or "").strip()
        if not attack or not basename:
            continue
        out.setdefault(attack, {})[basename] = dict(row)
    return out


def _paired_counts(
    poisoned_rows: Mapping[str, Mapping[str, Any]],
    defended_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Return (a, b, c, d) counts for a McNemar 2x2 table.

    a: both success, b: baseline-only success, c: defended-only success,
    d: neither success.
    """

    a = b = c = d = 0
    common = sorted(set(poisoned_rows) & set(defended_rows))
    for key in common:
        before = bool(poisoned_rows[key].get("success", False))
        after = bool(defended_rows[key].get("success", False))
        if before and after:
            a += 1
        elif before and not after:
            b += 1
        elif not before and after:
            c += 1
        else:
            d += 1
    return {"a": a, "b": b, "c": c, "d": d, "n_paired": len(common)}


def evaluate_defense_entry(entry: DefenseEntry, *, cfg: DefenseLeaderboardConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DefenseLeaderboardConfig()

    poisoned = load_json(entry.poisoned_external) if entry.poisoned_external is not None else None
    defended = load_json(entry.defended_external) if entry.defended_external is not None else None
    clean_before = load_json(entry.clean_before) if entry.clean_before is not None else None
    clean_after = load_json(entry.clean_after) if entry.clean_after is not None else None

    poisoned_summary = summarize_external_report(poisoned, confidence=cfg.confidence) if poisoned else None
    defended_summary = summarize_external_report(defended, confidence=cfg.confidence) if defended else None
    poisoned_counts = extract_counts(poisoned) if poisoned else {}
    defended_counts = extract_counts(defended) if defended else {}

    before_map = _as_map(clean_before, cfg)
    after_map = _as_map(clean_after, cfg)
    map_drop = None
    if before_map is not None and after_map is not None:
        map_drop = before_map - after_map

    # Per-attack comparison with Wilson CIs, absolute/relative delta, and
    # optional McNemar p-value when paired rows are available.
    per_attack: dict[str, Any] = {}
    paired_poisoned = _rows_by_attack(poisoned) if entry.use_paired_rows else {}
    paired_defended = _rows_by_attack(defended) if entry.use_paired_rows else {}
    attacks = sorted(
        set((poisoned_summary or {}).get("asr_matrix", {}))
        | set((defended_summary or {}).get("asr_matrix", {}))
    )
    warnings: list[str] = []
    blockers: list[str] = []
    for attack in attacks:
        p_matrix = (poisoned_summary or {}).get("asr_matrix", {})
        d_matrix = (defended_summary or {}).get("asr_matrix", {})
        p_asr = float(p_matrix.get(attack, 0.0))
        d_asr = float(d_matrix.get(attack, 0.0))
        row: dict[str, Any] = {
            "poisoned_asr": p_asr,
            "defended_asr": d_asr,
            "asr_absolute_reduction": p_asr - d_asr,
            "asr_relative_reduction": (p_asr - d_asr) / p_asr if p_asr > 0 else 0.0,
            "defended_wilson_ci": None,
            "poisoned_wilson_ci": None,
            "defended_zero_failure_upper_bound": None,
            "regression": d_asr > p_asr + cfg.max_per_attack_regression,
            "mcnemar": None,
        }
        if attack in poisoned_counts:
            s, n = poisoned_counts[attack]
            row["poisoned_wilson_ci"] = wilson_interval(s, n, cfg.confidence).to_dict()
        if attack in defended_counts:
            s, n = defended_counts[attack]
            row["defended_wilson_ci"] = wilson_interval(s, n, cfg.confidence).to_dict()
            if s == 0:
                row["defended_zero_failure_upper_bound"] = zero_failure_upper_bound(n, cfg.confidence)
        if paired_poisoned.get(attack) and paired_defended.get(attack):
            pair = _paired_counts(paired_poisoned[attack], paired_defended[attack])
            pvalue = mcnemar_exact_pvalue(pair["b"], pair["c"])
            row["mcnemar"] = {
                **pair,
                "p_value": pvalue,
                "significant": pvalue <= cfg.min_paired_sig_alpha and pair["b"] > pair["c"],
                "defense_improved": pair["b"] > pair["c"],
            }
        if row["regression"]:
            blockers.append(f"{attack}: defended ASR {d_asr:.6g} > poisoned ASR {p_asr:.6g}")
        per_attack[attack] = row

    # Clean-accuracy (mAP) constraint.
    if map_drop is not None and map_drop > cfg.max_clean_map_drop:
        blockers.append(
            f"clean mAP50-95 drop {map_drop:.6g} > tolerance {cfg.max_clean_map_drop:.6g}"
        )
    elif map_drop is None:
        warnings.append("clean mAP before/after not supplied; OD-DER will skip the clean term")

    # OD-DER.
    #
    # BackdoorBench DER = clamp(0.5 * ((C-Acc_before - C-Acc_after)/2 + (ASR_before - ASR_after)/2)).
    # We keep the spirit - average of two normalized gains - but:
    #   1. adopt mAP50-95 instead of classification accuracy;
    #   2. compute ASR gain as mean across all attacks (not just the single
    #      target), so that leaving one attack worse is visible;
    #   3. clamp the final score to [0, 1] and emit ``accepted`` separately
    #      so a defense that improves the mean but regresses on one attack
    #      scores well on "average gain" yet fails acceptance.
    mean_poisoned_asr = _mean([row["poisoned_asr"] for row in per_attack.values()]) if per_attack else 0.0
    mean_defended_asr = _mean([row["defended_asr"] for row in per_attack.values()]) if per_attack else 0.0
    asr_gain = max(0.0, min(1.0, mean_poisoned_asr - mean_defended_asr))
    map_gain = 0.0
    if before_map is not None and after_map is not None and before_map > 0:
        # Defense should not degrade clean performance.  Gain is +1 if the
        # defense kept mAP intact, -1 if it halved it.  We clamp at 0 so a
        # tolerated drop does not penalize a successful ASR reduction twice.
        map_gain = max(0.0, 1.0 - (max(0.0, map_drop or 0.0) / max(1e-6, cfg.max_clean_map_drop)))
    od_der = 0.5 * (asr_gain + map_gain)

    accepted = not blockers
    return {
        "entry": entry.to_dict(),
        "accepted": accepted,
        "blockers": blockers,
        "warnings": warnings,
        "clean": {
            "map50_95_before": before_map,
            "map50_95_after": after_map,
            "map50_95_drop": map_drop,
            "tolerance": cfg.max_clean_map_drop,
        },
        "poisoned": poisoned_summary,
        "defended": defended_summary,
        "per_attack": per_attack,
        "aggregate": {
            "mean_poisoned_asr": mean_poisoned_asr,
            "mean_defended_asr": mean_defended_asr,
            "mean_asr_reduction": mean_poisoned_asr - mean_defended_asr,
            "max_defended_asr": max((row["defended_asr"] for row in per_attack.values()), default=0.0),
            "asr_gain": asr_gain,
            "map_gain": map_gain,
            "od_der": od_der,
            "n_attacks_with_regression": sum(1 for row in per_attack.values() if row["regression"]),
            "n_attacks_with_paired_sig_improvement": sum(
                1
                for row in per_attack.values()
                if row["mcnemar"] and row["mcnemar"].get("significant")
            ),
        },
        "config": asdict(cfg),
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Leaderboard building
# ---------------------------------------------------------------------------


def _ranking_key(row: Mapping[str, Any]) -> tuple[int, float, float, float]:
    """Sort defended entries: accepted first, then by OD-DER, then max defended ASR, then mAP drop."""

    aggregate = row.get("aggregate") or {}
    clean = row.get("clean") or {}
    return (
        0 if row.get("accepted") else 1,
        -float(aggregate.get("od_der", 0.0)),
        float(aggregate.get("max_defended_asr", 1.0)),
        float(clean.get("map50_95_drop") or 0.0),
    )


def build_defense_leaderboard(
    entries: Sequence[DefenseEntry],
    *,
    cfg: DefenseLeaderboardConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DefenseLeaderboardConfig()
    rows = [evaluate_defense_entry(entry, cfg=cfg) for entry in entries]
    ranked = sorted(rows, key=_ranking_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "n_entries": len(ranked),
        "n_accepted": sum(1 for row in ranked if row.get("accepted")),
        "rows": ranked,
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Markdown rendering and writers
# ---------------------------------------------------------------------------


def _fmt_ci(interval: Mapping[str, Any] | None) -> str:
    if not interval:
        return "-"
    return "{s}/{t} = {rate:.4f} [{low:.4f}, {high:.4f}]".format(
        s=int(interval.get("successes", 0)),
        t=int(interval.get("total", 0)),
        rate=float(interval.get("rate", 0.0)),
        low=float(interval.get("low", 0.0)),
        high=float(interval.get("high", 0.0)),
    )


def _fmt_mcnemar(mcnemar: Mapping[str, Any] | None) -> str:
    if not mcnemar:
        return "-"
    p = mcnemar.get("p_value")
    sig = "yes" if mcnemar.get("significant") else "no"
    return f"b={mcnemar.get('b')} c={mcnemar.get('c')} p={p:.4g} sig={sig}"


def render_defense_leaderboard_markdown(leaderboard: Mapping[str, Any]) -> str:
    cfg = leaderboard.get("config") or {}
    lines: list[str] = ["# T0 OD Defense Leaderboard", ""]
    lines.append(f"- entries: `{leaderboard.get('n_entries')}`")
    lines.append(f"- accepted: `{leaderboard.get('n_accepted')}`")
    lines.append(
        f"- thresholds: clean mAP drop `<= {cfg.get('max_clean_map_drop')}`, "
        f"per-attack regression `<= {cfg.get('max_per_attack_regression')}`, "
        f"paired sig alpha `<= {cfg.get('min_paired_sig_alpha')}`"
    )
    lines.append(
        "- OD-DER is a strict-dominance extension of BackdoorBench DER "
        "(Wu et al., 2022, https://arxiv.org/abs/2206.12654): a defense is "
        "accepted only if no per-attack regression, clean mAP drop within "
        "tolerance, and every attack with paired rows reports either "
        "McNemar-significant improvement or an already-low defended ASR."
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| rank | name | defense | poisoned id | accepted | OD-DER | max defended ASR | mean ASR reduction | mAP drop | paired sig attacks |"
    )
    lines.append("|---:|---|---|---|:-:|---:|---:|---:|---:|---:|")
    for row in leaderboard.get("rows", []):
        entry = row.get("entry") or {}
        agg = row.get("aggregate") or {}
        clean = row.get("clean") or {}
        accepted = "PASS" if row.get("accepted") else "FAIL"
        lines.append(
            "| `{rank}` | `{name}` | `{defense}` | `{pid}` | {accepted} | "
            "`{oder:.4f}` | `{max_def:.4f}` | `{mean_red:.4f}` | "
            "`{drop}` | `{sig}` |".format(
                rank=row.get("rank"),
                name=entry.get("name"),
                defense=entry.get("defense"),
                pid=entry.get("poisoned_model_id"),
                accepted=accepted,
                oder=float(agg.get("od_der", 0.0)),
                max_def=float(agg.get("max_defended_asr", 0.0)),
                mean_red=float(agg.get("mean_asr_reduction", 0.0)),
                drop=(f"{float(clean.get('map50_95_drop')):+.4f}" if clean.get("map50_95_drop") is not None else "-"),
                sig=int(agg.get("n_attacks_with_paired_sig_improvement", 0)),
            )
        )

    lines.append("")
    lines.append("## Per-Attack Detail")
    for row in leaderboard.get("rows", []):
        entry = row.get("entry") or {}
        lines.append("")
        lines.append(f"### rank {row.get('rank')}: `{entry.get('name')}` (defense=`{entry.get('defense')}`)")
        if row.get("blockers"):
            lines.append("")
            lines.append("Blockers:")
            for msg in row["blockers"]:
                lines.append(f"- {msg}")
        if row.get("warnings"):
            lines.append("")
            lines.append("Warnings:")
            for msg in row["warnings"]:
                lines.append(f"- {msg}")
        lines.append("")
        lines.append(
            "| attack | poisoned ASR | defended ASR | abs red | rel red | "
            "defended Wilson CI | McNemar | regression |"
        )
        lines.append("|---|---:|---:|---:|---:|---|---|:-:|")
        for attack, detail in (row.get("per_attack") or {}).items():
            lines.append(
                "| `{attack}` | `{p:.4f}` | `{d:.4f}` | `{abs_red:+.4f}` | "
                "`{rel:+.4f}` | {ci} | {mc} | `{reg}` |".format(
                    attack=attack,
                    p=float(detail.get("poisoned_asr", 0.0)),
                    d=float(detail.get("defended_asr", 0.0)),
                    abs_red=float(detail.get("asr_absolute_reduction", 0.0)),
                    rel=float(detail.get("asr_relative_reduction", 0.0)),
                    ci=_fmt_ci(detail.get("defended_wilson_ci")),
                    mc=_fmt_mcnemar(detail.get("mcnemar")),
                    reg="yes" if detail.get("regression") else "no",
                )
            )
    return "\n".join(lines) + "\n"


def write_defense_leaderboard(out_dir: str | Path, leaderboard: Mapping[str, Any]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "t0_defense_leaderboard.json"
    md_path = out / "T0_DEFENSE_LEADERBOARD.md"
    json_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_defense_leaderboard_markdown(leaderboard), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Loaders: read a manifest file (JSON/YAML) to drive the CLI.
# ---------------------------------------------------------------------------


def load_entries_from_manifest(path: str | Path) -> list[DefenseEntry]:
    """Load a leaderboard manifest.

    The manifest is a mapping with an ``entries`` list; each item provides
    paths to the poisoned/defended external reports and the clean-before /
    clean-after metric files.  Minimal required fields: ``name``,
    ``poisoned_model_id``, ``defense``.
    """

    data: Mapping[str, Any]
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Cannot parse manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Manifest {path} must be a mapping")
    rows = data.get("entries") or []
    out: list[DefenseEntry] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        out.append(
            DefenseEntry(
                name=str(item.get("name") or item.get("defense") or "unnamed"),
                poisoned_model_id=str(item.get("poisoned_model_id") or "unknown_poisoned"),
                defense=str(item.get("defense") or "unknown_defense"),
                poisoned_external=item.get("poisoned_external"),
                defended_external=item.get("defended_external"),
                clean_before=item.get("clean_before"),
                clean_after=item.get("clean_after"),
                use_paired_rows=bool(item.get("use_paired_rows", True)),
                notes=str(item.get("notes") or ""),
            )
        )
    return out


__all__ = [
    "DefenseLeaderboardConfig",
    "DefenseEntry",
    "mcnemar_exact_pvalue",
    "evaluate_defense_entry",
    "build_defense_leaderboard",
    "render_defense_leaderboard_markdown",
    "write_defense_leaderboard",
    "load_entries_from_manifest",
]
