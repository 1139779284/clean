"""T0 detox ablation planner and orchestrator.

This module is the glue that connects contribution 1 (Hybrid-PURIFY-OD
manifests produced by ``scripts/hybrid_purify_detox_yolo.py``) to
contribution 3 (CFRC defense certificates produced by
``scripts/t0_defense_certificate.py``).  Its job is to answer three
questions a paper reviewer always asks:

1. Which ablation runs are expected and what commands produce them?
2. Which runs exist on disk, which are missing, and when were they run?
3. Once all paired (poisoned baseline, defended) artifacts exist, what are
   the CFRC DefenseEntry rows that should go into the final evidence
   table?

The planner is intentionally read-only and idempotent.  It never launches
training by itself; it prints the exact commands and writes a runbook plus
the CFRC manifest whose entries map 1:1 to the ablation's arm labels.  GPU
owners run the commands; the planner reports completion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AblationArm:
    """One row in the ablation grid.

    The canonical contribution-1 ablation is
    ``static_lambda`` vs ``lagrangian_lambda`` on the same poisoned model
    and the same external hard suite.  Additional arms can disable
    individual loss terms or feature modules via ``config_overrides``,
    which are emitted as ``--key value`` CLI flags on the Hybrid-PURIFY
    command (underscores become dashes, booleans emit a bare flag).
    """

    name: str
    use_lagrangian_controller: bool
    extra_cli: tuple[str, ...] = ()
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra_cli"] = list(self.extra_cli)
        d["config_overrides"] = dict(self.config_overrides)
        return d


@dataclass(frozen=True)
class DetoxAblationSpec:
    """One ``(poisoned baseline, defense ablation)`` cell."""

    poisoned_model_id: str
    poisoned_model_path: str
    teacher_model: str | None
    data_yaml: str
    images: str
    labels: str
    target_classes: tuple[str, ...]
    external_eval_roots: tuple[str, ...]
    external_replay_roots: tuple[str, ...] = ()
    hybrid_config: str | None = None
    imgsz: int = 640
    batch: int = 8
    cycles: int = 3
    phase_epochs: int = 2
    feature_epochs: int = 2
    recovery_epochs: int = 2
    device: str = "0"
    max_allowed_external_asr: float = 0.10
    max_map_drop: float = 0.03
    # Optional sizing knobs for smoke or constrained-GPU runs.  When None
    # the existing Hybrid-PURIFY defaults apply (no CLI flag emitted).
    max_images: int | None = None
    eval_max_images: int | None = None
    external_eval_max_images_per_attack: int | None = None
    external_replay_max_images_per_attack: int | None = None
    selection_max_map_drop: float | None = None
    no_pre_prune: bool = False
    out_root: str = "runs/t0_detox_ablation"
    extra_cli: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class PlannedRun:
    arm: str
    out_dir: str
    poisoned_external: str
    defended_external: str
    clean_before: str
    clean_after: str
    hybrid_manifest: str
    train_command: tuple[str, ...]
    exists: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_command"] = list(self.train_command)
        d["exists"] = dict(self.exists)
        return d


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------


def _q(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text


def _cli_flag(value: Any, flag: str, default: Any) -> list[str]:
    if value is None or value == default:
        return []
    return [flag, str(value)]


def build_arm_train_command(
    *,
    spec: DetoxAblationSpec,
    arm: AblationArm,
    python: str = "python",
) -> list[str]:
    """Construct the Hybrid-PURIFY training command for one arm."""

    out_dir = Path(spec.out_root) / arm.name
    cmd: list[str] = [
        python,
        "scripts/hybrid_purify_detox_yolo.py",
    ]
    if spec.hybrid_config:
        cmd.extend(["--config", spec.hybrid_config])
    cmd.extend([
        "--model",
        spec.poisoned_model_path,
        "--images",
        spec.images,
        "--labels",
        spec.labels,
        "--data-yaml",
        spec.data_yaml,
        "--target-classes",
        *spec.target_classes,
        "--external-eval-roots",
        *spec.external_eval_roots,
        "--out",
        str(out_dir),
        "--imgsz",
        str(spec.imgsz),
        "--batch",
        str(spec.batch),
        "--cycles",
        str(spec.cycles),
        "--phase-epochs",
        str(spec.phase_epochs),
        "--feature-epochs",
        str(spec.feature_epochs),
        "--recovery-epochs",
        str(spec.recovery_epochs),
        "--max-allowed-external-asr",
        str(spec.max_allowed_external_asr),
        "--max-map-drop",
        str(spec.max_map_drop),
        "--device",
        str(spec.device),
    ])
    if spec.teacher_model:
        cmd.extend(["--teacher-model", spec.teacher_model])
    if spec.external_replay_roots:
        cmd.append("--external-replay-roots")
        cmd.extend(list(spec.external_replay_roots))
    # Optional sizing flags emitted only when the spec sets them.  This keeps
    # the generated command backward compatible with older HybridPurifyConfig
    # defaults.
    if spec.max_images is not None:
        cmd.extend(["--max-images", str(spec.max_images)])
    if spec.eval_max_images is not None:
        cmd.extend(["--eval-max-images", str(spec.eval_max_images)])
    if spec.external_eval_max_images_per_attack is not None:
        cmd.extend([
            "--external-eval-max-images-per-attack",
            str(spec.external_eval_max_images_per_attack),
        ])
    if spec.external_replay_max_images_per_attack is not None:
        cmd.extend([
            "--external-replay-max-images-per-attack",
            str(spec.external_replay_max_images_per_attack),
        ])
    if spec.selection_max_map_drop is not None:
        cmd.extend(["--selection-max-map-drop", str(spec.selection_max_map_drop)])
    if spec.no_pre_prune:
        cmd.append("--no-pre-prune")
    # Arm-level config overrides: each key/value becomes ``--key value`` with
    # underscores replaced by dashes.  Bool True emits a bare flag; bool False
    # is skipped.  Lists/tuples emit ``--key v1 v2 ...``.
    for key, value in dict(arm.config_overrides or {}).items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
            continue
        cmd.extend([flag, str(value)])
    if arm.use_lagrangian_controller:
        cmd.append("--use-lagrangian-controller")
    cmd.extend(list(spec.extra_cli))
    cmd.extend(list(arm.extra_cli))
    return cmd


# ---------------------------------------------------------------------------
# Hybrid-PURIFY manifest -> CFRC entry conversion
# ---------------------------------------------------------------------------


def _manifest_paths(manifest: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Extract poisoned-baseline and defended external/clean paths.

    Hybrid-PURIFY writes ``before_eval`` for the poisoned baseline and
    ``best`` for the currently accepted checkpoint.  Depending on whether
    any cycle improved on the baseline, the two ``external_json`` pointers
    can either be distinct or point at the same file (baseline rolled
    forward unchanged).  For CFRC this is fine: bootstrap on identical
    success vectors produces Δ=0 with zero CI width, correctly reporting
    "defense did not improve this attack".

    Clean metrics are embedded inline; we persist them to files when they
    are not already present on disk so CFRC can load them by path.
    """

    before = manifest.get("before_eval") or {}
    best = manifest.get("best") or {}
    cycles = manifest.get("cycles") or []
    # Preferred source: ``before_eval.external_json`` / ``best.external_json``
    # if Hybrid-PURIFY emitted them.  ``before_eval`` historically does not
    # carry the JSON pointer directly, so we also accept the ``best`` pointer
    # at cycle 0 as the poisoned baseline when the purifier never replaced
    # the initial checkpoint.  When later cycles exist, their ``external_json``
    # is used as the defended pointer.
    poisoned_external = (
        before.get("external_json")
        or (best.get("external_json") if best.get("cycle") in (0, None) else None)
        or ""
    )
    # Canonical on-disk location for the baseline external report, written
    # by ``_evaluate_all(..., tag="00_before")``.  Use it as a final fallback
    # so the planner can always hand CFRC a poisoned-baseline pointer even
    # when the manifest does not spell it out (the common case after the
    # defender accepted a phase candidate during the first cycle).
    if not poisoned_external:
        canonical_before = out_dir / "eval_00_before_external" / "external_hard_suite_asr.json"
        if canonical_before.exists():
            poisoned_external = str(canonical_before)
    defended_external = (
        best.get("external_json")
        or (cycles[-1].get("external_json") if cycles else None)
        or poisoned_external
    )
    clean_before_path = out_dir / "clean_before.json"
    clean_after_path = out_dir / "clean_after.json"
    if before.get("clean_metrics"):
        clean_before_path.write_text(
            json.dumps(before["clean_metrics"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if best.get("clean_metrics"):
        clean_after_path.write_text(
            json.dumps(best["clean_metrics"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "poisoned_external": str(poisoned_external),
        "defended_external": str(defended_external),
        "clean_before": str(clean_before_path),
        "clean_after": str(clean_after_path),
        "hybrid_manifest": str(out_dir / "hybrid_purify_manifest.json"),
    }


def hybrid_manifest_to_defense_entry(
    *,
    arm_name: str,
    poisoned_model_id: str,
    hybrid_manifest_path: str | Path,
) -> dict[str, Any] | None:
    """Return a CFRC-compatible entry dict for one completed arm.

    Returns ``None`` if the manifest is missing ``external_json`` paths that
    would make the CFRC rows unresolvable.
    """

    manifest_path = Path(hybrid_manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = _manifest_paths(manifest, manifest_path.parent)
    if not paths["poisoned_external"] or not paths["defended_external"]:
        return None
    return {
        "name": arm_name,
        "poisoned_model_id": poisoned_model_id,
        "defense": arm_name,
        "poisoned_external": paths["poisoned_external"],
        "defended_external": paths["defended_external"],
        "clean_before": paths["clean_before"],
        "clean_after": paths["clean_after"],
        "use_paired_rows": True,
        "notes": f"Generated by t0.ablation_plan from {manifest_path}",
    }


# ---------------------------------------------------------------------------
# Plan orchestration
# ---------------------------------------------------------------------------


def _existence(paths: Mapping[str, str]) -> dict[str, bool]:
    return {key: bool(value) and Path(value).exists() for key, value in paths.items()}


def plan_runs(
    spec: DetoxAblationSpec,
    arms: Sequence[AblationArm],
    *,
    python: str = "python",
) -> list[PlannedRun]:
    """Plan all ablation runs for ``spec``.

    For each arm the planner constructs the full Hybrid-PURIFY training
    command and records the paths where the poisoned/defended external
    reports and clean metrics are expected to land.  Existence is probed;
    missing files are reported as ``exists[<key>] = False``.
    """

    runs: list[PlannedRun] = []
    for arm in arms:
        out_dir = Path(spec.out_root) / arm.name
        hybrid_manifest = out_dir / "hybrid_purify_manifest.json"
        # When the manifest already exists, use its reported paths; otherwise
        # record the conventional expected paths so the plan is actionable.
        if hybrid_manifest.exists():
            data = json.loads(hybrid_manifest.read_text(encoding="utf-8"))
            paths = _manifest_paths(data, out_dir)
        else:
            final_cycle_tag = f"eval_cycle_{int(spec.cycles):02d}_external"
            paths = {
                "poisoned_external": str(out_dir / "eval_00_before_external" / "external_hard_suite_asr.json"),
                "defended_external": str(out_dir / final_cycle_tag / "external_hard_suite_asr.json"),
                "clean_before": str(out_dir / "clean_before.json"),
                "clean_after": str(out_dir / "clean_after.json"),
                "hybrid_manifest": str(hybrid_manifest),
            }
        command = build_arm_train_command(spec=spec, arm=arm, python=python)
        runs.append(
            PlannedRun(
                arm=arm.name,
                out_dir=str(out_dir),
                poisoned_external=paths["poisoned_external"],
                defended_external=paths["defended_external"],
                clean_before=paths["clean_before"],
                clean_after=paths["clean_after"],
                hybrid_manifest=paths["hybrid_manifest"],
                train_command=tuple(command),
                exists=_existence(paths),
            )
        )
    return runs


def build_cfrc_manifest(
    *,
    poisoned_model_id: str,
    runs: Sequence[PlannedRun],
) -> dict[str, Any]:
    """Build the CFRC defense manifest from completed plan runs."""

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run in runs:
        entry = hybrid_manifest_to_defense_entry(
            arm_name=run.arm,
            poisoned_model_id=poisoned_model_id,
            hybrid_manifest_path=run.hybrid_manifest,
        )
        if entry is None:
            skipped.append(
                {
                    "arm": run.arm,
                    "reason": "hybrid_purify_manifest.json not found or missing external_json paths",
                    "expected_manifest": run.hybrid_manifest,
                }
            )
            continue
        entries.append(entry)
    return {
        "entries": entries,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(_q(part) for part in cmd)


def render_runbook_markdown(
    *,
    spec: DetoxAblationSpec,
    arms: Sequence[AblationArm],
    runs: Sequence[PlannedRun],
) -> str:
    lines: list[str] = ["# T0 Detox Ablation Runbook", ""]
    lines.append(f"- poisoned model: `{spec.poisoned_model_id}`")
    lines.append(f"- poisoned checkpoint: `{spec.poisoned_model_path}`")
    lines.append(f"- data yaml: `{spec.data_yaml}`")
    lines.append(f"- external eval roots: `{', '.join(spec.external_eval_roots)}`")
    lines.append(f"- out root: `{spec.out_root}`")
    lines.append(f"- arms: `{', '.join(a.name for a in arms)}`")
    lines.append("")
    lines.append(
        "Each arm is one Hybrid-PURIFY-OD training run against the same "
        "poisoned model and external hard suite.  The difference between "
        "arms is encoded in the CLI command below."
    )
    lines.append("")
    for run, arm in zip(runs, arms):
        lines.append(f"## Arm `{arm.name}`")
        lines.append("")
        if arm.notes:
            lines.append(f"Notes: {arm.notes}")
            lines.append("")
        lines.append("```powershell")
        lines.append(_fmt_cmd(run.train_command))
        lines.append("```")
        lines.append("")
        lines.append("Expected artifacts:")
        lines.append(f"- hybrid manifest: `{run.hybrid_manifest}`")
        lines.append(f"- poisoned external: `{run.poisoned_external}`")
        lines.append(f"- defended external: `{run.defended_external}`")
        lines.append(f"- clean before: `{run.clean_before}`")
        lines.append(f"- clean after: `{run.clean_after}`")
        lines.append("")
        lines.append("Currently on disk:")
        for key, ok in run.exists.items():
            mark = "YES" if ok else "MISSING"
            lines.append(f"- {key}: {mark}")
        lines.append("")
    lines.append("## Final CFRC certification")
    lines.append("")
    lines.append(
        "Once every arm's hybrid manifest has been produced, run "
        "``pixi run t0-defense-certificate --manifest <cfrc_manifest>``.  The "
        "planner emits that manifest to the runbook's ``out_dir`` as "
        "``cfrc_manifest.json``.  Arms with missing artifacts are recorded "
        "under ``skipped`` so the final table is reproducible."
    )
    return "\n".join(lines) + "\n"


def write_ablation_plan(
    out_dir: str | Path,
    *,
    spec: DetoxAblationSpec,
    arms: Sequence[AblationArm],
    runs: Sequence[PlannedRun],
    cfrc_manifest: Mapping[str, Any],
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_json = out / "t0_detox_ablation_plan.json"
    plan_json.write_text(
        json.dumps(
            {
                "spec": asdict(spec),
                "arms": [arm.to_dict() for arm in arms],
                "runs": [run.to_dict() for run in runs],
                "cfrc_manifest": dict(cfrc_manifest),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    runbook_md = out / "T0_DETOX_ABLATION_RUNBOOK.md"
    runbook_md.write_text(
        render_runbook_markdown(spec=spec, arms=arms, runs=runs),
        encoding="utf-8",
    )
    cfrc_json = out / "cfrc_manifest.json"
    cfrc_json.write_text(
        json.dumps(dict(cfrc_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "plan": plan_json,
        "runbook": runbook_md,
        "cfrc_manifest": cfrc_json,
    }


# ---------------------------------------------------------------------------
# Convenience: default contribution-1 ablation arms.
# ---------------------------------------------------------------------------


def default_contribution_1_arms() -> tuple[AblationArm, AblationArm]:
    """Return the canonical ``static_lambda`` vs ``lagrangian_lambda`` arms."""

    return (
        AblationArm(
            name="static_lambda",
            use_lagrangian_controller=False,
            notes="Baseline: static per-phase lambdas (pre-Lagrangian behaviour).",
        ),
        AblationArm(
            name="lagrangian_lambda",
            use_lagrangian_controller=True,
            notes=(
                "Contribution 1: adaptive per-attack Lagrangian lambda driven "
                "by the external ASR matrix at each cycle."
            ),
        ),
    )


__all__ = [
    "AblationArm",
    "DetoxAblationSpec",
    "PlannedRun",
    "build_arm_train_command",
    "hybrid_manifest_to_defense_entry",
    "plan_runs",
    "build_cfrc_manifest",
    "render_runbook_markdown",
    "write_ablation_plan",
    "default_contribution_1_arms",
]
