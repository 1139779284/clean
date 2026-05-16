from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import load_json, summarize_external_report


@dataclass(frozen=True)
class PoisonMatrixEvidenceConfig:
    expected_attacks: tuple[str, ...] = ("badnet_oga_corner", "semantic_cleanlabel", "wanet_oga")
    expected_seeds: tuple[int, ...] = ()
    expected_poison_rates: tuple[float, ...] = ()
    min_primary_asr: float = 0.20
    min_usable_asr: float = 0.05
    require_weights: bool = True
    require_report: bool = True
    require_any_strong: bool = True
    require_full_factorial: bool = False
    full_factorial_cell_acceptance: str = "strong"
    poison_rate_tolerance: float = 1e-9


@dataclass
class PoisonModelEvidenceEntry:
    attack: str
    run: str
    poison_rate: float | None = None
    seed: int | None = None
    epochs: int | None = None
    weights: str | None = None
    report: str | None = None
    weights_exists: bool = False
    report_exists: bool = False
    intended_attack_asr: float = 0.0
    max_asr: float = 0.0
    mean_asr: float = 0.0
    status: str = "missing"
    blocked_reasons: list[str] = field(default_factory=list)
    asr_matrix: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    normalized = {_norm(key): float(value) for key, value in matrix.items()}
    for alias in _attack_aliases(attack):
        if alias in normalized:
            return normalized[alias]
    for key, value in normalized.items():
        if key.startswith(_norm(attack)) or _norm(attack).startswith(key):
            return value
    return 0.0


def _seed_matches(entry: PoisonModelEvidenceEntry, seed: int | None) -> bool:
    return seed is None or entry.seed == int(seed)


def _poison_rate_matches(entry: PoisonModelEvidenceEntry, poison_rate: float | None, tolerance: float) -> bool:
    return poison_rate is None or (
        entry.poison_rate is not None and abs(float(entry.poison_rate) - float(poison_rate)) <= float(tolerance)
    )


def _factor_labels(values: Sequence[Any], *, default: Any = None) -> list[Any]:
    return list(values) if values else [default]


def _cell_is_accepted(entry: PoisonModelEvidenceEntry, cfg: PoisonMatrixEvidenceConfig) -> bool:
    mode = str(cfg.full_factorial_cell_acceptance or "strong").strip().lower()
    if mode == "present":
        return entry.report_exists and (entry.weights_exists or not cfg.require_weights)
    if mode == "usable":
        return entry.status in {"weak", "strong"}
    return entry.status == "strong"


def _load_entries_from_summary(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = summary.get("entries")
    if isinstance(entries, list):
        return [dict(item) for item in entries if isinstance(item, Mapping)]
    return []


def _entry_from_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    run = path.parent.name
    attack = run.split("_pr", 1)[0] if "_pr" in run else run
    row: dict[str, Any] = {"attack": attack, "run": run, "report": str(path)}
    match = re.search(r"_pr(?P<rate>\d{4})_seed(?P<seed>\d+)", run)
    if match:
        row["poison_rate"] = int(match.group("rate")) / 10000.0
        row["seed"] = int(match.group("seed"))
    return row


def evaluate_poison_matrix_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    root: str | Path = ".",
    cfg: PoisonMatrixEvidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PoisonMatrixEvidenceConfig()
    base = Path(root)
    evaluated: list[PoisonModelEvidenceEntry] = []

    for raw in entries:
        attack = str(raw.get("attack") or raw.get("attack_name") or raw.get("run") or "unknown")
        run = str(raw.get("run") or raw.get("name") or attack)
        report = raw.get("report")
        weights = raw.get("weights") or raw.get("model")
        report_path = (base / str(report)).resolve() if report else None
        weights_path = (base / str(weights)).resolve() if weights else None

        blocked: list[str] = []
        report_exists = bool(report_path and report_path.exists())
        weights_exists = bool(weights_path and weights_path.exists())
        if cfg.require_report and not report_exists:
            blocked.append("missing ASR report")
        if cfg.require_weights and not weights_exists:
            blocked.append("missing weights")

        summary: dict[str, Any] = {}
        if report_exists and report_path is not None:
            summary = summarize_external_report(load_json(report_path))
        matrix = dict(summary.get("asr_matrix") or {})
        intended = _intended_asr(matrix, attack)
        max_asr = float(summary.get("max_asr", 0.0) or 0.0)
        mean_asr = float(summary.get("mean_asr", 0.0) or 0.0)

        if intended < float(cfg.min_usable_asr):
            blocked.append(f"intended attack ASR {intended:.6g} < usable {float(cfg.min_usable_asr):.6g}")
            status = "blocked"
        elif intended < float(cfg.min_primary_asr):
            status = "weak"
        else:
            status = "strong"
        if blocked:
            status = "blocked"

        evaluated.append(
            PoisonModelEvidenceEntry(
                attack=attack,
                run=run,
                poison_rate=float(raw["poison_rate"]) if raw.get("poison_rate") is not None else None,
                seed=int(raw["seed"]) if raw.get("seed") is not None else None,
                epochs=int(raw["epochs"]) if raw.get("epochs") is not None else None,
                weights=str(weights) if weights else None,
                report=str(report) if report else None,
                weights_exists=weights_exists,
                report_exists=report_exists,
                intended_attack_asr=float(intended),
                max_asr=max_asr,
                mean_asr=mean_asr,
                status=status,
                blocked_reasons=blocked,
                asr_matrix={str(k): float(v) for k, v in matrix.items()},
            )
        )

    expected = list(cfg.expected_attacks)
    expected_seeds = _factor_labels(cfg.expected_seeds)
    expected_poison_rates = _factor_labels(cfg.expected_poison_rates)
    coverage: dict[str, dict[str, Any]] = {}
    for attack in expected:
        aliases = set(_attack_aliases(attack))
        matches = [entry for entry in evaluated if _norm(entry.attack) in aliases]
        strong = [entry for entry in matches if entry.status == "strong"]
        weak = [entry for entry in matches if entry.status == "weak"]
        best = max(matches, key=lambda item: item.intended_attack_asr, default=None)
        missing_cells: list[dict[str, Any]] = []
        if cfg.require_full_factorial:
            for seed in expected_seeds:
                for poison_rate in expected_poison_rates:
                    cell_accepted = [
                        entry
                        for entry in matches
                        if _seed_matches(entry, seed)
                        and _poison_rate_matches(entry, poison_rate, cfg.poison_rate_tolerance)
                        and _cell_is_accepted(entry, cfg)
                    ]
                    if not cell_accepted:
                        missing_cells.append({"seed": seed, "poison_rate": poison_rate})
        coverage[attack] = {
            "complete": not bool(missing_cells),
            "strong": bool(strong) and not missing_cells,
            "has_any_strong": bool(strong),
            "weak": bool(weak),
            "n_entries": len(matches),
            "best_run": best.run if best else None,
            "best_intended_attack_asr": best.intended_attack_asr if best else 0.0,
            "missing_cells": missing_cells,
            "blocked": (not bool(strong)) or bool(missing_cells),
        }

    blocked_reasons = []
    cell_acceptance = str(cfg.full_factorial_cell_acceptance or "strong").strip().lower()
    for attack, row in coverage.items():
        if cfg.require_any_strong and not row["has_any_strong"]:
            blocked_reasons.append(f"missing strong poison model for {attack}")
        if row["missing_cells"]:
            blocked_reasons.append(f"missing full-factorial {cell_acceptance} cells for {attack}: {len(row['missing_cells'])}")
    status = "passed" if not blocked_reasons else "blocked"
    return {
        "status": status,
        "accepted": status == "passed",
        "blocked_reasons": blocked_reasons,
        "coverage": coverage,
        "entries": [entry.to_dict() for entry in evaluated],
        "config": asdict(cfg),
    }


def build_poison_matrix_evidence(
    *,
    summary_json: str | Path | None = None,
    report_paths: Sequence[str | Path] = (),
    root: str | Path = ".",
    cfg: PoisonMatrixEvidenceConfig | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if summary_json:
        entries.extend(_load_entries_from_summary(load_json(summary_json)))
    entries.extend(_entry_from_report(path) for path in report_paths)
    return evaluate_poison_matrix_entries(entries, root=root, cfg=cfg)


def write_poison_matrix_evidence(out_dir: str | Path, evidence: Mapping[str, Any]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "t0_poison_matrix_evidence.json"
    md_path = out / "T0_POISON_MATRIX_EVIDENCE.md"
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# T0 Poison Matrix Evidence", ""]
    lines.append(f"- status: `{evidence.get('status')}`")
    for reason in evidence.get("blocked_reasons") or []:
        lines.append(f"- blocked: {reason}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    cell_acceptance = ((evidence.get("config") or {}).get("full_factorial_cell_acceptance") or "strong")
    lines.append(f"- full-factorial cell acceptance: `{cell_acceptance}`")
    lines.append("")
    lines.append("| attack | complete | has strong | best intended ASR | missing cells | best run |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for attack, row in (evidence.get("coverage") or {}).items():
        lines.append(
            f"| `{attack}` | `{bool(row.get('complete'))}` | `{bool(row.get('has_any_strong'))}` | "
            f"`{float(row.get('best_intended_attack_asr', 0.0)):.6f}` | "
            f"`{len(row.get('missing_cells') or [])}` | `{row.get('best_run')}` |"
        )
    lines.append("")
    lines.append("## Entries")
    lines.append("")
    lines.append("| status | attack | run | intended ASR | max ASR | weights |")
    lines.append("|---|---|---|---:|---:|---|")
    for entry in evidence.get("entries") or []:
        lines.append(
            f"| `{entry.get('status')}` | `{entry.get('attack')}` | `{entry.get('run')}` | "
            f"`{float(entry.get('intended_attack_asr', 0.0)):.6f}` | `{float(entry.get('max_asr', 0.0)):.6f}` | `{entry.get('weights')}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
