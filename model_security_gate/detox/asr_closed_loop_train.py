from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from model_security_gate.detox.asr_aware_dataset import (
    ASRAwareDatasetConfig,
    AttackTransformConfig,
    build_asr_aware_yolo_dataset,
    class_names_from_yaml_or_mapping,
    default_attack_suite,
)
from model_security_gate.detox.asr_regression import ASRRegressionConfig, run_asr_regression_for_yolo, write_asr_regression_outputs
from model_security_gate.detox.common import find_ultralytics_weight
from model_security_gate.detox.external_hard_suite import (
    ExternalAttackDataset,
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    attack_score_lookup,
    discover_external_attack_datasets,
    run_external_hard_suite_for_yolo,
    score_for_attack_name,
    write_external_hard_suite_outputs,
)
from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.utils.io import resolve_class_ids, write_json


@dataclass
class ClosedLoopPhase:
    name: str
    attacks: Sequence[AttackTransformConfig] = field(default_factory=tuple)
    epochs: int = 2
    clean_repeat: int = 2
    attack_repeat: int = 1
    lr0: float = 2e-5
    weight_decay: float = 7e-4
    replay_external: bool = True
    train_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRClosedLoopConfig:
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None
    workers: int = 0
    seed: int = 42
    cycles: int = 4
    max_allowed_external_asr: float = 0.10
    max_allowed_internal_asr: float = 0.10
    max_map_drop: float = 0.03
    min_map50_95: float | None = None
    val_fraction: float = 0.15
    max_images: int = 0
    eval_max_images: int = 0
    external_eval_roots: Sequence[str] = field(default_factory=tuple)
    external_replay_roots: Sequence[str] = field(default_factory=tuple)
    external_eval_max_images_per_attack: int = 0
    external_replay_max_images_per_attack: int = 250
    external_oda_success_mode: str = "localized_any_recalled"
    # Phase schedule knobs.
    base_clean_repeat: int = 2
    recovery_clean_repeat: int = 4
    base_attack_repeat: int = 1
    max_attack_repeat: int = 5
    adaptive_boost: float = 3.0
    active_asr_threshold: float = 0.08
    top_k_attacks_per_cycle: int = 3
    phase_epochs: int = 3
    recovery_epochs: int = 2
    clean_anchor_every_cycle: bool = False
    clean_recovery_every_cycle: bool = False
    lr0: float = 2e-5
    recovery_lr0: float = 1e-5
    weight_decay: float = 7e-4
    attack_specs: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())
    include_internal_asr: bool = True
    use_external_replay: bool = True
    external_replay_failed_only: bool = True
    rollback_on_external_regression: bool = True
    rollback_on_no_improvement: bool = True
    max_external_asr_regression: float = 0.005
    max_external_mean_asr_regression: float = 0.02
    min_selection_improvement: float = 1e-4
    stop_on_pass: bool = True


def _eval_clean_yolo(
    model_path: str | Path,
    data_yaml: str | Path,
    imgsz: int,
    batch: int,
    device: str | int | None = None,
    workers: int = 0,
) -> Dict[str, Any] | None:
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        kwargs: Dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": int(workers),
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        metrics = model.val(**kwargs)
        return {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
    except Exception as exc:  # noqa: BLE001 - clean metrics are optional but recorded.
        return {"error": str(exc)}


def _map_drop(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> float | None:
    if not before or not after or "map50_95" not in before or "map50_95" not in after:
        return None
    try:
        return float(before["map50_95"]) - float(after["map50_95"])
    except Exception:
        return None


def _max_asr(result: Mapping[str, Any] | None) -> float:
    try:
        return float(((result or {}).get("summary") or {}).get("max_asr") or 0.0)
    except Exception:
        return 0.0


def _mean_asr(result: Mapping[str, Any] | None) -> float:
    try:
        return float(((result or {}).get("summary") or {}).get("mean_asr") or 0.0)
    except Exception:
        return 0.0


def _selection_score(
    external_asr: float,
    internal_asr: float,
    map_drop: float | None,
    cfg: ASRClosedLoopConfig,
    external_mean_asr: float | None = None,
) -> float:
    # External hard suite dominates selection because internal regression can be self-consistent.
    score = 1.20 * float(external_asr) + 0.25 * float(external_mean_asr or 0.0) + 0.35 * float(internal_asr)
    if map_drop is not None and map_drop > float(cfg.max_map_drop):
        score += 8.0 * (float(map_drop) - float(cfg.max_map_drop))
    return float(score)


def _attack_groups(specs: Sequence[AttackTransformConfig]) -> Dict[str, List[AttackTransformConfig]]:
    groups = {"oga": [], "oda": [], "semantic": [], "wanet": []}
    for spec in specs:
        goal = str(spec.goal).lower()
        kind = str(spec.kind).lower()
        if goal == "oda":
            groups["oda"].append(spec)
        elif "wanet" in kind or "warp" in kind or "wanet" in spec.name.lower():
            groups["wanet"].append(spec)
        elif goal in {"semantic", "all", "both"} or "semantic" in spec.name.lower():
            groups["semantic"].append(spec)
        else:
            groups["oga"].append(spec)
    return groups


def _repeat_for_score(score: float, cfg: ASRClosedLoopConfig) -> int:
    if score <= 0:
        return int(cfg.base_attack_repeat)
    # Increase attack replay only when the external score is above the desired zone.
    scaled = max(0.0, (float(score) - float(cfg.active_asr_threshold)) / max(float(cfg.active_asr_threshold), 1e-6))
    repeat = int(round(float(cfg.base_attack_repeat) + float(cfg.adaptive_boost) * scaled))
    return max(int(cfg.base_attack_repeat), min(int(cfg.max_attack_repeat), repeat))


def _trigger_preserving_kwargs() -> Dict[str, Any]:
    return {
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "label_smoothing": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
    }


def _build_phase_plan(
    attack_specs: Sequence[AttackTransformConfig],
    hard_scores: Mapping[str, float],
    cfg: ASRClosedLoopConfig,
) -> List[ClosedLoopPhase]:
    groups = _attack_groups(attack_specs)
    scored_specs: List[tuple[AttackTransformConfig, float]] = []
    for spec in attack_specs:
        scored_specs.append((spec, score_for_attack_name(hard_scores, spec.name, kind=spec.kind, goal=spec.goal)))
    scored_specs.sort(key=lambda x: x[1], reverse=True)
    top_names = {spec.name for spec, score in scored_specs[: max(1, int(cfg.top_k_attacks_per_cycle))] if score >= cfg.active_asr_threshold}

    phases: List[ClosedLoopPhase] = []
    # Clean anchor first: avoids immediately overfitting to hard negatives.
    if cfg.clean_anchor_every_cycle or not hard_scores:
        phases.append(
            ClosedLoopPhase(
                name="clean_anchor",
                attacks=(),
                epochs=max(1, int(cfg.recovery_epochs)),
                clean_repeat=int(cfg.recovery_clean_repeat),
                attack_repeat=0,
                lr0=float(cfg.recovery_lr0),
                replay_external=False,
                train_kwargs={"mosaic": 0.5, "mixup": 0.05, "copy_paste": 0.05, "erasing": 0.20, "label_smoothing": 0.03},
            )
        )

    selected_groups: List[tuple[str, List[AttackTransformConfig], float]] = []
    for group_name in ["oga", "oda", "semantic", "wanet"]:
        specs = groups[group_name]
        if not specs:
            continue
        # Activate group if any member is hard, or in early cycles when no external history exists.
        selected = [s for s in specs if s.name in top_names]
        if not selected and not hard_scores:
            selected = specs
        if not selected:
            continue
        group_score = max([score_for_attack_name(hard_scores, s.name, kind=s.kind, goal=s.goal) for s in selected] + [0.0])
        selected_groups.append((group_name, selected, group_score))

    # Run the hardest external failure mode first. With detector backdoors this
    # matters: OGA hardening can raise recall/FP behavior and accidentally make
    # ODA or semantic failures worse if it always runs first.
    selected_groups.sort(key=lambda item: item[2], reverse=True)

    for group_name, selected, group_score in selected_groups:
        attack_repeat = _repeat_for_score(group_score, cfg)
        if group_name == "oga":
            clean_repeat = max(int(cfg.base_clean_repeat), int(round(attack_repeat * 0.5)))
        elif group_name == "oda":
            clean_repeat = max(int(cfg.base_clean_repeat), int(attack_repeat))
        else:
            clean_repeat = max(int(cfg.base_clean_repeat), int(round(attack_repeat * 1.0)))
        if group_name == "oda":
            kwargs = _trigger_preserving_kwargs()
        elif group_name == "semantic":
            kwargs = {"mosaic": 0.15, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.05, "hsv_h": 0.02, "hsv_s": 0.25, "hsv_v": 0.20, "label_smoothing": 0.0}
        elif group_name == "wanet":
            kwargs = _trigger_preserving_kwargs()
        else:
            kwargs = _trigger_preserving_kwargs()
        phases.append(
            ClosedLoopPhase(
                name=f"{group_name}_hardening",
                attacks=selected,
                epochs=max(1, int(cfg.phase_epochs)),
                clean_repeat=clean_repeat,
                attack_repeat=attack_repeat,
                lr0=float(cfg.lr0),
                weight_decay=float(cfg.weight_decay),
                replay_external=True,
                train_kwargs=kwargs,
            )
        )

    if cfg.clean_recovery_every_cycle:
        phases.append(
            ClosedLoopPhase(
                name="clean_recovery",
                attacks=(),
                epochs=max(1, int(cfg.recovery_epochs)),
                clean_repeat=int(cfg.recovery_clean_repeat),
                attack_repeat=0,
                lr0=float(cfg.recovery_lr0),
                weight_decay=float(cfg.weight_decay),
                replay_external=False,
                train_kwargs={"mosaic": 0.7, "mixup": 0.10, "copy_paste": 0.08, "erasing": 0.25, "label_smoothing": 0.03},
            )
        )
    return phases


def _build_phase_dataset(
    phase: ClosedLoopPhase,
    cycle: int,
    output_dir: Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    names: Mapping[int, str],
    target_ids: Sequence[int],
    cfg: ASRClosedLoopConfig,
    replay_datasets: Sequence[ExternalAttackDataset],
    failure_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    phase_dir = output_dir / f"01_cycle_{cycle:02d}_{phase.name}_dataset"
    yaml_path = build_asr_aware_yolo_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=phase_dir,
        class_names=names,
        cfg=ASRAwareDatasetConfig(
            val_fraction=cfg.val_fraction,
            seed=cfg.seed + cycle,
            include_clean=True,
            include_clean_repeat=max(1, int(phase.clean_repeat)),
            include_attack_repeat=max(0, int(phase.attack_repeat)),
            max_images=cfg.max_images,
            target_class_ids=target_ids,
            attacks=list(phase.attacks),
        ),
    )
    replay_stats: Dict[str, Any] = {"added": 0, "skipped": 0, "by_attack": {}}
    if cfg.use_external_replay and phase.replay_external and replay_datasets:
        replay_stats = append_external_replay_samples(
            output_dataset_dir=phase_dir,
            attack_datasets=replay_datasets,
            target_class_ids=target_ids,
            selected_attack_names=[a.name for a in phase.attacks],
            max_images_per_attack=int(cfg.external_replay_max_images_per_attack),
            split="train",
            seed=cfg.seed + cycle,
            failure_rows=failure_rows,
            failure_only=bool(cfg.external_replay_failed_only),
        )
    write_json(phase_dir / "phase_manifest.json", {"phase": asdict(phase), "replay_stats": replay_stats, "data_yaml": str(yaml_path)})
    return yaml_path


def _evaluate_all(
    model: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    cfg: ASRClosedLoopConfig,
    external_eval_cfg: ExternalHardSuiteConfig | None,
    output_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if external_eval_cfg and (external_eval_cfg.roots or external_eval_cfg.attacks):
        external = run_external_hard_suite_for_yolo(model, data_yaml=data_yaml, target_classes=target_classes, cfg=external_eval_cfg, device=cfg.device)
        ext_dir = output_dir / f"eval_{tag}_external"
        ext_json, ext_rows = write_external_hard_suite_outputs(external, ext_dir)
        out["external"] = external
        out["external_json"] = str(ext_json)
        out["external_rows"] = str(ext_rows)
    if cfg.include_internal_asr:
        internal_cfg = ASRRegressionConfig(imgsz=cfg.imgsz, max_images=cfg.eval_max_images, attacks=cfg.attack_specs)
        internal = run_asr_regression_for_yolo(model, images_dir=images_dir, labels_dir=labels_dir, data_yaml=data_yaml, target_classes=target_classes, cfg=internal_cfg, device=cfg.device)
        int_dir = output_dir / f"eval_{tag}_internal"
        int_json, int_rows = write_asr_regression_outputs(internal, int_dir)
        out["internal"] = internal
        out["internal_json"] = str(int_json)
        out["internal_rows"] = str(int_rows)
    out["clean_metrics"] = _eval_clean_yolo(model, data_yaml, cfg.imgsz, cfg.batch, cfg.device, cfg.workers)
    return out


def _combined_scores(evals: Mapping[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for key in ["external", "internal"]:
        scores.update(attack_score_lookup(evals.get(key)))
    return scores


def _candidate_item(
    *,
    cycle: int,
    model: str | Path,
    evals: Mapping[str, Any],
    clean_before: Mapping[str, Any] | None,
    cfg: ASRClosedLoopConfig,
    phase_record: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    external_asr = _max_asr(evals.get("external")) if evals.get("external") else _max_asr(evals.get("internal"))
    internal_asr = _max_asr(evals.get("internal"))
    clean_after = evals.get("clean_metrics")
    drop = _map_drop(clean_before, clean_after)
    external_mean = _mean_asr(evals.get("external")) if evals.get("external") else None
    score = _selection_score(external_asr, internal_asr, drop, cfg, external_mean_asr=external_mean)
    return {
        "cycle": cycle,
        "model": str(model),
        "external_max_asr": float(external_asr),
        "external_mean_asr": external_mean,
        "internal_max_asr": float(internal_asr),
        "internal_mean_asr": _mean_asr(evals.get("internal")) if evals.get("internal") else None,
        "clean_metrics": clean_after,
        "map_drop": drop,
        "selection_score": score,
        "passes_external_asr": external_asr <= float(cfg.max_allowed_external_asr),
        "passes_internal_asr": internal_asr <= float(cfg.max_allowed_internal_asr),
        "passes_map": (drop is None) or (drop <= float(cfg.max_map_drop)),
        "eval_paths": {k: v for k, v in evals.items() if k.endswith("_json") or k.endswith("_rows")},
        "phase_record": dict(phase_record or {}),
    }


def run_asr_closed_loop_detox_yolo(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    output_dir: str | Path,
    cfg: ASRClosedLoopConfig | None = None,
) -> Dict[str, Any]:
    """External-hard-suite closed-loop ASR detox.

    This is stricter than ``asr_aware_train``: checkpoint selection is driven by
    external hard-suite ASR first, then internal ASR, then clean mAP. Training is
    split into OGA/ODA/semantic/WaNet phases plus clean recovery, and can replay
    external hard-suite samples during the matching phase to close the internal
    vs external distribution gap.
    """
    cfg = cfg or ASRClosedLoopConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names_from_yaml_or_mapping(data_yaml)
    target_ids = resolve_class_ids(names, target_classes)
    if not target_ids:
        raise ValueError("Closed-loop ASR detox requires explicit target_classes")

    external_eval_cfg = ExternalHardSuiteConfig(
        roots=tuple(cfg.external_eval_roots or ()),
        max_images_per_attack=int(cfg.external_eval_max_images_per_attack),
        imgsz=cfg.imgsz,
        oda_success_mode=cfg.external_oda_success_mode,
        seed=cfg.seed,
    )
    replay_roots = tuple(cfg.external_replay_roots or cfg.external_eval_roots or ())
    replay_datasets = discover_external_attack_datasets(replay_roots)

    manifest: Dict[str, Any] = {
        "input_model": str(model_path),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml": str(data_yaml),
        "target_classes": [str(x) for x in target_classes],
        "target_class_ids": target_ids,
        "config": {**asdict(cfg), "attack_specs": [asdict(a) for a in cfg.attack_specs]},
        "external_eval_roots": list(cfg.external_eval_roots or []),
        "external_replay_roots": list(replay_roots),
        "replay_datasets": [asdict(d) for d in replay_datasets],
        "cycles": [],
        "best": None,
        "status": "running",
        "warnings": [],
    }
    if not cfg.external_eval_roots:
        manifest["warnings"].append("No external_eval_roots were provided; selection falls back to internal ASR and is less reliable.")
    write_json(output_dir / "asr_closed_loop_detox_manifest.json", manifest)

    current_model = Path(model_path)
    clean_before = _eval_clean_yolo(current_model, data_yaml, cfg.imgsz, cfg.batch, cfg.device, cfg.workers)
    manifest["clean_before"] = clean_before
    pre_eval = _evaluate_all(current_model, images_dir, labels_dir, data_yaml, target_classes, cfg, external_eval_cfg, output_dir, tag="00_before")
    manifest["before_eval"] = {k: v for k, v in pre_eval.items() if k.endswith("_json") or k == "clean_metrics"}
    hard_scores = _combined_scores(pre_eval)
    last_external_rows: Sequence[Mapping[str, Any]] | None = (pre_eval.get("external") or {}).get("rows")
    best_item: Dict[str, Any] | None = _candidate_item(
        cycle=0,
        model=current_model,
        evals=pre_eval,
        clean_before=clean_before,
        cfg=cfg,
        phase_record={"phase": "baseline"},
    )
    manifest["best"] = best_item
    accepted_model = Path(best_item["model"])
    accepted_hard_scores = dict(hard_scores)
    accepted_external_rows: Sequence[Mapping[str, Any]] | None = last_external_rows

    for cycle in range(1, int(cfg.cycles) + 1):
        current_model = accepted_model
        phases = _build_phase_plan(cfg.attack_specs, accepted_hard_scores, cfg)
        cycle_record: Dict[str, Any] = {"cycle": cycle, "phases": [asdict(p) for p in phases]}
        for phase_idx, phase in enumerate(phases, 1):
            phase_yaml = _build_phase_dataset(phase, cycle, output_dir, images_dir, labels_dir, names, target_ids, cfg, replay_datasets, failure_rows=accepted_external_rows)
            project = output_dir / f"02_cycle_{cycle:02d}_phase_{phase_idx:02d}_{phase.name}_train"
            train_kwargs = dict(phase.train_kwargs)
            train_counterfactual_finetune(
                base_model=current_model,
                data_yaml=phase_yaml,
                output_project=project,
                name="closed_loop",
                imgsz=cfg.imgsz,
                epochs=int(phase.epochs),
                batch=cfg.batch,
                device=cfg.device,
                workers=cfg.workers,
                lr0=float(phase.lr0),
                weight_decay=float(phase.weight_decay),
                **train_kwargs,
            )
            current_model = find_ultralytics_weight(project, "closed_loop", prefer="best")
            cycle_record.setdefault("trained_phases", []).append({"phase": phase.name, "model": str(current_model), "data_yaml": str(phase_yaml)})

        evals = _evaluate_all(current_model, images_dir, labels_dir, data_yaml, target_classes, cfg, external_eval_cfg, output_dir, tag=f"cycle_{cycle:02d}")
        item = _candidate_item(cycle=cycle, model=current_model, evals=evals, clean_before=clean_before, cfg=cfg, phase_record=cycle_record)
        previous_best_external = float((best_item or {}).get("external_max_asr") or 0.0)
        previous_best_external_mean = (best_item or {}).get("external_mean_asr")
        external_mean_regressed = (
            item.get("external_mean_asr") is not None
            and previous_best_external_mean is not None
            and float(item["external_mean_asr"]) > float(previous_best_external_mean) + float(cfg.max_external_mean_asr_regression)
        )
        improved = best_item is None or item["selection_score"] < float(best_item["selection_score"]) - float(cfg.min_selection_improvement)
        should_rollback = bool(cfg.rollback_on_external_regression) and (
            item["external_max_asr"] > previous_best_external + float(cfg.max_external_asr_regression)
            or external_mean_regressed
            or (bool(cfg.rollback_on_no_improvement) and not improved)
        )
        if should_rollback:
            item["rolled_back"] = True
            item["rollback_to"] = str(accepted_model)
            if item["external_max_asr"] > previous_best_external + float(cfg.max_external_asr_regression):
                item["rollback_reason"] = "external_max_asr_regressed"
            elif external_mean_regressed:
                item["rollback_reason"] = "external_mean_asr_regressed"
            else:
                item["rollback_reason"] = "no_selection_improvement"
            current_model = accepted_model
            improved = False
        else:
            item["rolled_back"] = False
        manifest["cycles"].append(item)
        if improved:
            best_item = item
            manifest["best"] = item
            accepted_model = Path(item["model"])
            accepted_hard_scores = _combined_scores(evals)
            accepted_external_rows = (evals.get("external") or {}).get("rows") or accepted_external_rows
        write_json(output_dir / "asr_closed_loop_detox_manifest.json", manifest)
        if cfg.stop_on_pass and item["passes_external_asr"] and item["passes_internal_asr"] and item["passes_map"]:
            manifest["status"] = "passed_early"
            break

    if best_item is None:
        manifest["status"] = "failed_no_checkpoint"
    else:
        manifest["final_model"] = best_item["model"]
        manifest["status"] = "passed" if best_item["passes_external_asr"] and best_item["passes_internal_asr"] and best_item["passes_map"] else "failed_external_asr_or_map"
    write_json(output_dir / "asr_closed_loop_detox_manifest.json", manifest)
    return manifest
