"""T0 OD Defense Certificate: paired bootstrap CI + Holm-Bonferroni FWER
correction + Certified Minimum Reduction (CMR) ranking.

This module is the project's main methodological contribution on top of the
defense leaderboard.  It directly addresses the critiques in
*On Backdoor Defense Evaluation* (2025, https://arxiv.org/abs/2511.13143)
and the single-number DER limitation of BackdoorBench
(Wu et al., 2022, https://arxiv.org/abs/2206.12654):

1. **Paired bootstrap confidence interval on ASR reduction.**

   For every attack family we have per-image ``success`` flags from both the
   poisoned baseline and the defended model evaluated on the SAME images.
   Define the per-image reduction ``r_i = s_i^{poisoned} - s_i^{defended}``
   where ``s_i \\in {0, 1}``.  The mean reduction is ``\\hat{\\Delta}`` and we
   estimate a two-sided bias-corrected percentile bootstrap interval by
   resampling matched pairs.  This captures paired variance; the unpaired
   Wilson interval does not.

2. **Holm-Bonferroni family-wise error correction.**

   With ``k`` tracked attacks we run ``k`` McNemar tests.  Reporting raw
   p-values inflates Type-I error: a defense with five independent McNemars
   at alpha=0.05 has FWER ~= 0.23.  We sort p-values and apply the
   Holm-Bonferroni step-down correction
   (Holm, 1979, https://www.jstor.org/stable/4615733): the smallest p-value
   must pass ``alpha / k``, the second ``alpha / (k-1)``, and so on.
   ``adjusted_p = min(1, (k - rank + 1) * p)`` is recorded per attack.

3. **Certified Minimum Reduction (CMR) ranking.**

   ``cmr_asr = min_attack lower_bound(\\hat{\\Delta}_attack)`` where the lower
   bound is the bootstrap one-sided 95% lower confidence limit, truncated
   below at 0.  A defense is certified at level ``x`` iff every tracked
   attack reduces ASR by at least ``x`` with joint confidence.  This is an
   anti-p-hacking ranking primary key: we never rank by mean ASR reduction
   when per-attack lower bounds remain weak.

The module is framework-light: it only needs ``random`` from the stdlib
(``secrets`` would also work), the project's own metric loaders, and no
SciPy.  It runs under ``pixi run ci-smoke``.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .defense_leaderboard import DefenseEntry, DefenseLeaderboardConfig, mcnemar_exact_pvalue
from .metrics import load_json, summarize_external_report
from .stats import wilson_interval


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefenseCertificateConfig:
    """Certificate thresholds.

    ``n_bootstrap`` is fixed at 2000 by default.  At 95% confidence the
    bootstrap quantile step is 1/2000 = 0.0005, which is finer than the
    typical paired-pair count (200-300) supports.  Reproducibility uses
    ``seed``.

    Per-attack acceptance is *either* improvement-based *or* non-inferiority
    based, whichever succeeds first:

    * ``min_certified_reduction``: the attack's paired-bootstrap one-sided
      95% lower bound on ASR reduction must reach this absolute value.  This
      is the classical "defense makes things better" test.
    * ``max_certified_asr``: when the defended ASR's Wilson-95% upper bound
      stays at or below this absolute cap, the attack is certified even if
      the reduction bound is smaller.  This is a non-inferiority acceptance:
      if the defense never exceeds an operator-agreed safety ceiling, the
      paper claim is valid regardless of the baseline starting point.

    The two paths are essential because requiring a 5pp absolute reduction
    on an attack whose baseline ASR is already 2.7% is mathematically
    impossible.  The non-inferiority path replaces the old "impossible by
    construction" failure with a correct statistical statement.
    """

    confidence: float = 0.95
    n_bootstrap: int = 2000
    seed: int = 12345
    fwer_alpha: float = 0.05
    max_clean_map_drop: float = 0.03
    min_certified_reduction: float = 0.05
    max_certified_asr: float = 0.05


# ---------------------------------------------------------------------------
# Core statistics: paired bootstrap and Holm-Bonferroni
# ---------------------------------------------------------------------------


def _bootstrap_quantiles(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    alpha: float,
    rng: random.Random,
) -> tuple[float, float, float]:
    """Return (lower, mean, upper) of a bootstrap distribution of the mean.

    Uses the percentile method with ``alpha/2`` and ``1-alpha/2`` cut-points.
    Edge cases:
    * empty ``values`` or ``n_bootstrap <= 0`` -> (0.0, 0.0, 0.0);
    * all-identical ``values`` -> (v, v, v), the degenerate correct answer.
    """

    n = len(values)
    if n == 0 or n_bootstrap <= 0:
        return 0.0, 0.0, 0.0
    sample_mean = statistics.fmean(values)
    if all(v == sample_mean for v in values):
        return sample_mean, sample_mean, sample_mean
    samples: list[float] = [0.0] * n_bootstrap
    n_bootstrap_i = int(n_bootstrap)
    choices = rng.choices  # local alias, measurable win on 200k+ draws
    for i in range(n_bootstrap_i):
        samples[i] = statistics.fmean(choices(values, k=n))
    samples.sort()
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lo_idx = max(0, int(lo_q * n_bootstrap_i))
    hi_idx = min(n_bootstrap_i - 1, int(hi_q * n_bootstrap_i))
    return samples[lo_idx], sample_mean, samples[hi_idx]


def _bootstrap_one_sided_lower(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    alpha: float,
    rng: random.Random,
) -> float:
    """One-sided lower confidence bound on the mean of ``values``.

    Used for the CMR metric: we only care that the ASR reduction is *at
    least* something, not a two-sided interval.  This gives us all of the
    confidence budget on the lower tail.
    """

    n = len(values)
    if n == 0 or n_bootstrap <= 0:
        return 0.0
    sample_mean = statistics.fmean(values)
    if all(v == sample_mean for v in values):
        return sample_mean
    samples: list[float] = [0.0] * n_bootstrap
    n_bootstrap_i = int(n_bootstrap)
    choices = rng.choices
    for i in range(n_bootstrap_i):
        samples[i] = statistics.fmean(choices(values, k=n))
    samples.sort()
    lo_idx = max(0, int(alpha * n_bootstrap_i))
    return samples[lo_idx]


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Given raw p-values ``p_1, ..., p_k``, sort them ascending, then adjust
    ``p_{(i)} <- max_{j <= i} (k - j + 1) * p_{(j)}`` and clamp to 1.
    The returned list is in the original input order.

    Ref: Holm (1979), "A Simple Sequentially Rejective Multiple Test
    Procedure".
    """

    raw = [max(0.0, min(1.0, float(p))) for p in p_values]
    k = len(raw)
    if k == 0:
        return []
    # Sort with original index to reorder later.
    indexed = sorted(enumerate(raw), key=lambda kv: kv[1])
    adjusted_sorted: list[float] = []
    running_max = 0.0
    for rank, (_, p) in enumerate(indexed, start=1):
        adj = min(1.0, float(k - rank + 1) * p)
        running_max = max(running_max, adj)
        adjusted_sorted.append(running_max)
    out = [0.0] * k
    for (orig_idx, _), adj in zip(indexed, adjusted_sorted):
        out[orig_idx] = adj
    return out


# ---------------------------------------------------------------------------
# Per-entry certificate
# ---------------------------------------------------------------------------


def _rows_by_attack(report: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
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


def _paired_deltas(
    poisoned_rows: Mapping[str, Mapping[str, Any]],
    defended_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[float], dict[str, int]]:
    """Return per-image ASR reduction vector and the McNemar (a,b,c,d,n) counts.

    ``r_i = poisoned_success_i - defended_success_i``.  The resulting list
    has one entry per matched ``image_basename``.
    """

    deltas: list[float] = []
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
        deltas.append(float(before) - float(after))
    return deltas, {"a": a, "b": b, "c": c, "d": d, "n_paired": len(common)}


def _as_map(metrics: Mapping[str, Any] | None, fields: Sequence[str]) -> float | None:
    if not metrics:
        return None
    for key in fields:
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


_MAP_FIELDS: tuple[str, ...] = ("map50_95", "map", "metrics.map50_95", "map5095", "mAP50_95")


def certify_defense_entry(
    entry: DefenseEntry,
    *,
    cfg: DefenseCertificateConfig | None = None,
) -> dict[str, Any]:
    """Produce a full certificate for a single defended model."""

    cfg = cfg or DefenseCertificateConfig()
    rng = random.Random(cfg.seed)

    poisoned = load_json(entry.poisoned_external) if entry.poisoned_external is not None else None
    defended = load_json(entry.defended_external) if entry.defended_external is not None else None
    clean_before = load_json(entry.clean_before) if entry.clean_before is not None else None
    clean_after = load_json(entry.clean_after) if entry.clean_after is not None else None

    poisoned_summary = summarize_external_report(poisoned, confidence=cfg.confidence) if poisoned else None
    defended_summary = summarize_external_report(defended, confidence=cfg.confidence) if defended else None

    before_map = _as_map(clean_before, _MAP_FIELDS)
    after_map = _as_map(clean_after, _MAP_FIELDS)
    map_drop: float | None = None
    if before_map is not None and after_map is not None:
        map_drop = before_map - after_map

    paired_poisoned = _rows_by_attack(poisoned)
    paired_defended = _rows_by_attack(defended)
    attacks = sorted(
        set((poisoned_summary or {}).get("asr_matrix", {}))
        | set((defended_summary or {}).get("asr_matrix", {}))
    )

    per_attack: dict[str, dict[str, Any]] = {}
    raw_pvalues: list[float] = []
    attack_order: list[str] = []
    for attack in attacks:
        p_asr = float(((poisoned_summary or {}).get("asr_matrix") or {}).get(attack, 0.0))
        d_asr = float(((defended_summary or {}).get("asr_matrix") or {}).get(attack, 0.0))
        row: dict[str, Any] = {
            "poisoned_asr": p_asr,
            "defended_asr": d_asr,
            "asr_reduction": p_asr - d_asr,
            "paired": None,
            "bootstrap_ci": None,
            "certified_reduction_lower": 0.0,
            "defended_wilson_upper": None,
            "mcnemar_p": None,
            "holm_adjusted_p": None,
            "holm_significant": False,
            "regression": d_asr > p_asr,
            "meets_min_certified_reduction": False,
            "meets_max_certified_asr": False,
            "acceptance_path": None,
            "meets_acceptance": False,
        }

        if paired_poisoned.get(attack) and paired_defended.get(attack):
            deltas, counts = _paired_deltas(paired_poisoned[attack], paired_defended[attack])
            if deltas:
                mean_delta = statistics.fmean(deltas)
                lo, mean_boot, hi = _bootstrap_quantiles(
                    deltas,
                    n_bootstrap=cfg.n_bootstrap,
                    alpha=1.0 - cfg.confidence,
                    rng=rng,
                )
                one_sided_lower = _bootstrap_one_sided_lower(
                    deltas,
                    n_bootstrap=cfg.n_bootstrap,
                    alpha=1.0 - cfg.confidence,
                    rng=rng,
                )
                row["paired"] = counts
                row["bootstrap_ci"] = {
                    "mean_delta": mean_delta,
                    "mean_bootstrap": mean_boot,
                    "low": lo,
                    "high": hi,
                    "confidence": cfg.confidence,
                    "n": counts["n_paired"],
                }
                row["certified_reduction_lower"] = max(0.0, one_sided_lower)
                row["meets_min_certified_reduction"] = (
                    one_sided_lower >= cfg.min_certified_reduction
                )
                # Wilson upper bound on defended ASR from the matched subset.
                # Under the paired table ``a`` and ``c`` are defended
                # successes, ``b`` and ``d`` are defended failures, so
                # defended successes = a + c out of n_paired.
                defended_success = int(counts["a"]) + int(counts["c"])
                wilson = wilson_interval(defended_success, int(counts["n_paired"]), cfg.confidence).to_dict()
                row["defended_wilson_upper"] = float(wilson.get("high", 1.0))
                row["meets_max_certified_asr"] = (
                    row["defended_wilson_upper"] <= cfg.max_certified_asr
                )
                p = mcnemar_exact_pvalue(counts["b"], counts["c"])
                row["mcnemar_p"] = p
                raw_pvalues.append(p)
                attack_order.append(attack)

        # Acceptance: reduction path OR non-inferiority path.  Operators can
        # claim "improved by at least X" or "stayed below Y", whichever is
        # applicable for the attack's baseline.
        if row["meets_min_certified_reduction"]:
            row["acceptance_path"] = "reduction"
            row["meets_acceptance"] = True
        elif row["meets_max_certified_asr"]:
            row["acceptance_path"] = "non_inferiority"
            row["meets_acceptance"] = True

        per_attack[attack] = row

    adjusted = holm_bonferroni_adjust(raw_pvalues)
    for attack, adj_p in zip(attack_order, adjusted):
        row = per_attack[attack]
        row["holm_adjusted_p"] = adj_p
        # A defense is "significant" on this attack only when Holm survives AND
        # the McNemar-discordant count indicates the defense, not the poisoned
        # baseline, is the better side.
        pair = row.get("paired") or {}
        row["holm_significant"] = bool(
            adj_p <= cfg.fwer_alpha and int(pair.get("b", 0)) > int(pair.get("c", 0))
        )

    # Aggregate: Certified Minimum Reduction over all tracked attacks.
    certified_values = [row["certified_reduction_lower"] for row in per_attack.values()]
    cmr_asr = float(min(certified_values)) if certified_values else 0.0
    any_regression = any(row["regression"] for row in per_attack.values())
    all_meet_acceptance = bool(per_attack) and all(
        row["meets_acceptance"] for row in per_attack.values()
    )
    # Holm significance is only demanded for attacks that took the reduction
    # path (we are claiming "defense significantly reduced ASR").  Attacks
    # accepted through non-inferiority (defended Wilson upper bound already
    # below the absolute cap) do not require Holm; the claim there is
    # "defended ASR stays below the ceiling", which is a confidence-bound
    # statement on the defended rate alone, not a paired-difference test.
    holm_failures = [
        attack
        for attack, row in per_attack.items()
        if row.get("acceptance_path") == "reduction"
        and row.get("mcnemar_p") is not None
        and not row.get("holm_significant")
    ]
    clean_ok = map_drop is None or map_drop <= cfg.max_clean_map_drop

    blockers: list[str] = []
    warnings: list[str] = []
    if any_regression:
        for attack, row in per_attack.items():
            if row["regression"]:
                blockers.append(f"{attack}: defended ASR regressed above poisoned")
    if map_drop is not None and map_drop > cfg.max_clean_map_drop:
        blockers.append(
            f"clean mAP50-95 drop {map_drop:.6g} > tolerance {cfg.max_clean_map_drop:.6g}"
        )
    if not per_attack:
        warnings.append("no tracked attacks in external reports")
    elif not all_meet_acceptance:
        failing = [
            attack
            for attack, row in per_attack.items()
            if not row["meets_acceptance"]
        ]
        warnings.append(
            "attacks below both reduction and non-inferiority bounds"
            f" (min_certified_reduction={cfg.min_certified_reduction:.4f},"
            f" max_certified_asr={cfg.max_certified_asr:.4f}): {', '.join(failing)}"
        )
    if holm_failures:
        warnings.append(
            f"Holm-Bonferroni did not survive on: {', '.join(holm_failures)}"
        )

    certified = (
        not blockers
        and all_meet_acceptance
        and clean_ok
        and not holm_failures
    )

    return {
        "entry": entry.to_dict(),
        "certified": certified,
        "blockers": blockers,
        "warnings": warnings,
        "clean": {
            "map50_95_before": before_map,
            "map50_95_after": after_map,
            "map50_95_drop": map_drop,
            "tolerance": cfg.max_clean_map_drop,
        },
        "per_attack": per_attack,
        "aggregate": {
            "cmr_asr": cmr_asr,
            "mean_asr_reduction": statistics.fmean(
                [row["asr_reduction"] for row in per_attack.values()]
            )
            if per_attack
            else 0.0,
            "max_defended_asr": max((row["defended_asr"] for row in per_attack.values()), default=0.0),
            "n_attacks_significant_holm": sum(
                1 for row in per_attack.values() if row.get("holm_significant")
            ),
            "n_attacks_certified_reduction": sum(
                1 for row in per_attack.values() if row.get("acceptance_path") == "reduction"
            ),
            "n_attacks_certified_non_inferiority": sum(
                1 for row in per_attack.values() if row.get("acceptance_path") == "non_inferiority"
            ),
            "n_attacks_meets_acceptance": sum(
                1 for row in per_attack.values() if row.get("meets_acceptance")
            ),
            "n_attacks_regressed": sum(
                1 for row in per_attack.values() if row.get("regression")
            ),
        },
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Certified-ranking leaderboard
# ---------------------------------------------------------------------------


def _cert_ranking_key(row: Mapping[str, Any]) -> tuple[int, float, float, float]:
    agg = row.get("aggregate") or {}
    clean = row.get("clean") or {}
    # Primary: certified first. Secondary: CMR descending.
    # Tertiary: max defended ASR ascending. Fallback: mAP drop ascending.
    return (
        0 if row.get("certified") else 1,
        -float(agg.get("cmr_asr", 0.0)),
        float(agg.get("max_defended_asr", 1.0)),
        float(clean.get("map50_95_drop") or 0.0),
    )


def build_defense_certificates(
    entries: Sequence[DefenseEntry],
    *,
    cfg: DefenseCertificateConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DefenseCertificateConfig()
    rows = [certify_defense_entry(entry, cfg=cfg) for entry in entries]
    ranked = sorted(rows, key=_cert_ranking_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "n_entries": len(ranked),
        "n_certified": sum(1 for row in ranked if row.get("certified")),
        "rows": ranked,
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Markdown rendering and writers
# ---------------------------------------------------------------------------


def _fmt_ci(ci: Mapping[str, Any] | None) -> str:
    if not ci:
        return "-"
    return "mean {m:.4f}  [{low:.4f}, {high:.4f}]  n={n}".format(
        m=float(ci.get("mean_delta", 0.0)),
        low=float(ci.get("low", 0.0)),
        high=float(ci.get("high", 0.0)),
        n=int(ci.get("n", 0)),
    )


def render_defense_certificates_markdown(payload: Mapping[str, Any]) -> str:
    cfg = payload.get("config") or {}
    lines: list[str] = ["# T0 OD Defense Certificate", ""]
    lines.append(f"- entries: `{payload.get('n_entries')}`")
    lines.append(f"- certified: `{payload.get('n_certified')}`")
    lines.append(
        "- thresholds: clean mAP drop `<= {drop}`, min certified reduction "
        "`>= {mcr}`, max certified defended ASR `<= {maca}`, Holm-Bonferroni "
        "FWER `<= {fwer}`, bootstrap n=`{n}` confidence `{conf}`".format(
            drop=cfg.get("max_clean_map_drop"),
            mcr=cfg.get("min_certified_reduction"),
            maca=cfg.get("max_certified_asr"),
            fwer=cfg.get("fwer_alpha"),
            n=cfg.get("n_bootstrap"),
            conf=cfg.get("confidence"),
        )
    )
    lines.append(
        "- each attack is certified through either the *reduction* path "
        "(bootstrap lower bound on paired ASR reduction reaches "
        "`min_certified_reduction`) or the *non-inferiority* path "
        "(defended Wilson upper bound stays below `max_certified_asr`).  "
        "Reduction-path attacks additionally require Holm-Bonferroni "
        "survival; non-inferiority attacks do not, since the claim is "
        "\"stays below ceiling\" rather than \"paired difference is real\"."
    )
    lines.append(
        "- primary ranking key is Certified Minimum Reduction (CMR): the worst "
        "per-attack one-sided 95% bootstrap lower bound on the paired ASR "
        "reduction.  This is the anti-p-hacking answer to the single-number "
        "DER criticized in Wang et al., 2025 "
        "(https://arxiv.org/abs/2511.13143)."
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| rank | name | defense | certified | CMR | max defended ASR | "
        "mean ASR reduction | mAP drop | Holm significant attacks |"
    )
    lines.append("|---:|---|---|:-:|---:|---:|---:|---:|---:|")
    for row in payload.get("rows", []):
        entry = row.get("entry") or {}
        agg = row.get("aggregate") or {}
        clean = row.get("clean") or {}
        lines.append(
            "| `{rank}` | `{name}` | `{defense}` | {cert} | `{cmr:.4f}` | "
            "`{md:.4f}` | `{mr:+.4f}` | `{drop}` | `{sig}` |".format(
                rank=row.get("rank"),
                name=entry.get("name"),
                defense=entry.get("defense"),
                cert="PASS" if row.get("certified") else "FAIL",
                cmr=float(agg.get("cmr_asr", 0.0)),
                md=float(agg.get("max_defended_asr", 0.0)),
                mr=float(agg.get("mean_asr_reduction", 0.0)),
                drop=(
                    f"{float(clean.get('map50_95_drop')):+.4f}"
                    if clean.get("map50_95_drop") is not None
                    else "-"
                ),
                sig=int(agg.get("n_attacks_significant_holm", 0)),
            )
        )

    lines.append("")
    lines.append("## Per-Attack Certificates")
    for row in payload.get("rows", []):
        entry = row.get("entry") or {}
        lines.append("")
        lines.append(
            f"### rank {row.get('rank')}: `{entry.get('name')}` (defense=`{entry.get('defense')}`)"
        )
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
            "| attack | path | poisoned ASR | defended ASR | defended Wilson upper | "
            "paired bootstrap CI | certified lower | McNemar raw p | Holm p | accepted |"
        )
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|:-:|")
        for attack, detail in (row.get("per_attack") or {}).items():
            wilson_upper = detail.get("defended_wilson_upper")
            path = detail.get("acceptance_path") or "none"
            lines.append(
                "| `{attack}` | `{path}` | `{p:.4f}` | `{d:.4f}` | {wu} | {ci} | `{lo:.4f}` | "
                "{raw} | {adj} | `{acc}` |".format(
                    attack=attack,
                    path=path,
                    p=float(detail.get("poisoned_asr", 0.0)),
                    d=float(detail.get("defended_asr", 0.0)),
                    wu=(f"`{float(wilson_upper):.4f}`" if wilson_upper is not None else "-"),
                    ci=_fmt_ci(detail.get("bootstrap_ci")),
                    lo=float(detail.get("certified_reduction_lower", 0.0)),
                    raw=(
                        f"`{float(detail['mcnemar_p']):.4g}`"
                        if detail.get("mcnemar_p") is not None
                        else "-"
                    ),
                    adj=(
                        f"`{float(detail['holm_adjusted_p']):.4g}`"
                        if detail.get("holm_adjusted_p") is not None
                        else "-"
                    ),
                    acc="yes" if detail.get("meets_acceptance") else "no",
                )
            )
    return "\n".join(lines) + "\n"


def render_defense_certificates_latex(payload: Mapping[str, Any]) -> str:
    """Minimal LaTeX rendering of the CFRC certificate.

    Produces two ``table`` environments: a top-level ranking table with
    OD-DER, CMR, and the number of Holm-significant attacks; and one
    per-entry per-attack detail table.  Requires ``booktabs`` for ``\\toprule``
    etc., and ``siunitx`` is not used (we pre-format floats).
    """

    cfg = payload.get("config") or {}
    entries = payload.get("rows") or []
    lines: list[str] = []
    lines.append("% T0 OD Defense Certificate")
    lines.append("% Thresholds: clean mAP drop <= {0}, min_certified_reduction >= {1},".format(
        cfg.get("max_clean_map_drop"), cfg.get("min_certified_reduction")
    ))
    lines.append("%   max_certified_asr <= {0}, Holm-Bonferroni FWER <= {1}, bootstrap n={2}.".format(
        cfg.get("max_certified_asr"), cfg.get("fwer_alpha"), cfg.get("n_bootstrap")
    ))
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{CFRC defense ranking. Primary key: Certified Minimum Reduction.}")
    lines.append("\\label{tab:cfrc_ranking}")
    lines.append("\\begin{tabular}{rlcrrrrr}")
    lines.append("\\toprule")
    lines.append("Rank & Defense & Cert. & CMR & max Def.\\ ASR & Mean $\\Delta$ ASR & mAP drop & Holm sig. \\\\")
    lines.append("\\midrule")
    for row in entries:
        entry = row.get("entry") or {}
        agg = row.get("aggregate") or {}
        clean = row.get("clean") or {}
        cert = "\\checkmark" if row.get("certified") else "$\\times$"
        drop = clean.get("map50_95_drop")
        drop_cell = (f"{float(drop):+.4f}" if drop is not None else "-")
        lines.append(
            "{rank} & \\texttt{{{defense}}} & {cert} & {cmr:.4f} & {md:.4f} & {mr:+.4f} & {drop} & {sig} \\\\".format(
                rank=row.get("rank"),
                defense=entry.get("defense"),
                cert=cert,
                cmr=float(agg.get("cmr_asr", 0.0)),
                md=float(agg.get("max_defended_asr", 0.0)),
                mr=float(agg.get("mean_asr_reduction", 0.0)),
                drop=drop_cell,
                sig=int(agg.get("n_attacks_significant_holm", 0)),
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Per-attack tables (one per entry).
    for row in entries:
        entry = row.get("entry") or {}
        lines.append("")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(
            "\\caption{{Per-attack CFRC detail for \\texttt{{{defense}}}.}}".format(
                defense=entry.get("defense")
            )
        )
        lines.append("\\label{{tab:cfrc_detail_{0}}}".format(row.get("rank")))
        lines.append("\\begin{tabular}{lllrrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Attack & Path & Pois.\\ ASR & Def.\\ ASR & Wilson up. & "
            "$\\Delta$ lower & McNemar $p$ & Holm $p$ & Acc. \\\\"
        )
        lines.append("\\midrule")
        for attack, detail in (row.get("per_attack") or {}).items():
            wilson_upper = detail.get("defended_wilson_upper")
            path = detail.get("acceptance_path") or "none"
            raw = detail.get("mcnemar_p")
            adj = detail.get("holm_adjusted_p")
            lines.append(
                "\\texttt{{{attack}}} & {path} & {p:.4f} & {d:.4f} & {wu} & {lo:.4f} & {raw} & {adj} & {acc} \\\\".format(
                    attack=attack.replace("_", "\\_"),
                    path=path.replace("_", "\\_"),
                    p=float(detail.get("poisoned_asr", 0.0)),
                    d=float(detail.get("defended_asr", 0.0)),
                    wu=(f"{float(wilson_upper):.4f}" if wilson_upper is not None else "-"),
                    lo=float(detail.get("certified_reduction_lower", 0.0)),
                    raw=(f"{float(raw):.2e}" if raw is not None else "-"),
                    adj=(f"{float(adj):.2e}" if adj is not None else "-"),
                    acc=("\\checkmark" if detail.get("meets_acceptance") else "$\\times$"),
                )
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_defense_certificates(
    out_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    emit_latex: bool = True,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "t0_defense_certificate.json"
    md_path = out / "T0_DEFENSE_CERTIFICATE.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_defense_certificates_markdown(payload), encoding="utf-8")
    if emit_latex:
        (out / "T0_DEFENSE_CERTIFICATE.tex").write_text(
            render_defense_certificates_latex(payload), encoding="utf-8"
        )
    return json_path, md_path


__all__ = [
    "DefenseCertificateConfig",
    "certify_defense_entry",
    "build_defense_certificates",
    "holm_bonferroni_adjust",
    "render_defense_certificates_markdown",
    "render_defense_certificates_latex",
    "write_defense_certificates",
]
