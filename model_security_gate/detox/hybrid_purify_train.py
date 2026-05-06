from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig, class_names_from_yaml_or_mapping, default_attack_suite
from model_security_gate.detox.asr_closed_loop_train import (
    ASRClosedLoopConfig,
    _build_phase_dataset,
    _build_phase_plan,
    _combined_scores,
    _evaluate_all,
    _map_drop,
    _max_asr,
    _mean_asr,
    _selection_score,
)
from model_security_gate.detox.external_hard_suite import ExternalHardSuiteConfig, discover_external_attack_datasets
from model_security_gate.detox.strong_train import StrongDetoxConfig as FeatureStrongDetoxConfig
from model_security_gate.detox.strong_train import run_strong_detox_training
from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.detox.common import find_ultralytics_weight
from model_security_gate.utils.io import resolve_class_ids, write_json


@dataclass
class HybridPurifyConfig:
    """External-suite driven feature-level detox for YOLO detectors.

    This is the strongest pipeline in this project. It combines:
    - external hard-suite replay and checkpoint selection;
    - phase-separated OGA/ODA/semantic/WaNet training;
    - PGBD-style prototype alignment and target-prototype suppression;
    - I-BAU-style adversarial unlearning;
    - NAD / feature / output distillation against a clean teacher;
    - clean recovery to protect mAP.

    It still requires real labels for final safety claims. If teacher_model is
    omitted, the pipeline falls back to a frozen copy of the starting model,
    which is weaker and should not be treated as a full safety proof.
    """

    imgsz: int = 640
    batch: int = 8
    device: str | int | None = None
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

    # Phase schedule. Keep phases short; selection is external-ASR driven.
    phase_epochs: int = 2
    recovery_epochs: int = 2
    feature_epochs: int = 2
    base_clean_repeat: int = 2
    recovery_clean_repeat: int = 5
    base_attack_repeat: int = 1
    max_attack_repeat: int = 5
    adaptive_boost: float = 3.0
    active_asr_threshold: float = 0.08
    top_k_attacks_per_cycle: int = 3

    lr: float = 2e-5
    recovery_lr: float = 1e-5
    weight_decay: float = 7e-4
    num_workers: int = 2
    amp: bool = False
    max_hook_layers: int = 6
    prototype_max_batches: int = 40

    # Pipeline switches.
    use_external_replay: bool = True
    include_internal_asr: bool = True
    stop_on_pass: bool = True
    run_feature_purifier: bool = True
    run_clean_recovery_finetune: bool = True
    trusted_teacher_required: bool = False

    # Optional conservative soft-pruning. Off by default because clean mAP is
    # already fragile in the user's experiments.
    run_pre_prune: bool = False
    pre_prune_top_k: int = 0

    attack_specs: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())


def _phase_feature_weights(phase_name: str) -> Dict[str, float]:
    low = phase_name.lower()
    # These weights deliberately separate failure modes rather than blending all
    # attacks into one ordinary fine-tune. The prototype_suppress term is the
    # detection adaptation of PGBD for target-absent OGA/semantic negatives.
    if "oga" in low:
        return {
            "lambda_task": 1.15,
            "lambda_adv": 0.40,
            "lambda_output_distill": 0.35,
            "lambda_feature_distill": 0.35,
            "lambda_nad": 0.45,
            "lambda_attention": 0.15,
            "lambda_prototype": 0.25,
            "lambda_proto_suppress": 0.65,
        }
    if "oda" in low:
        return {
            "lambda_task": 1.45,
            "lambda_adv": 0.25,
            "lambda_output_distill": 0.45,
            "lambda_feature_distill": 0.45,
            "lambda_nad": 0.50,
            "lambda_attention": 0.35,
            "lambda_prototype": 0.55,
            "lambda_proto_suppress": 0.10,
        }
    if "semantic" in low:
        return {
            "lambda_task": 1.10,
            "lambda_adv": 0.45,
            "lambda_output_distill": 0.45,
            "lambda_feature_distill": 0.65,
            "lambda_nad": 0.65,
            "lambda_attention": 0.25,
            "lambda_prototype": 0.55,
            "lambda_proto_suppress": 0.45,
        }
    if "wanet" in low or "warp" in low:
        return {
            "lambda_task": 1.10,
            "lambda_adv": 0.45,
            "lambda_output_distill": 0.60,
            "lambda_feature_distill": 0.70,
            "lambda_nad": 0.70,
            "lambda_attention": 0.20,
            "lambda_prototype": 0.35,
            "lambda_proto_suppress": 0.25,
        }
    # Clean anchor/recovery: keep output close to teacher and recover mAP.
    return {
        "lambda_task": 1.0,
        "lambda_adv": 0.08,
        "lambda_output_distill": 0.75,
        "lambda_feature_distill": 0.45,
        "lambda_nad": 0.55,
        "lambda_attention": 0.15,
        "lambda_prototype": 0.25,
        "lambda_proto_suppress": 0.05,
    }


def _run_feature_purifier_phase(
    model: str | Path,
    teacher_model: str | Path | None,
    data_yaml: str | Path,
    out_dir: str | Path,
    target_ids: Sequence[int],
    phase_name: str,
    cfg: HybridPurifyConfig,
) -> Path:
    weights = _phase_feature_weights(phase_name)
    fcfg = FeatureStrongDetoxConfig(
        model=str(model),
        data_yaml=str(data_yaml),
        out_dir=str(out_dir),
        teacher_model=str(teacher_model) if teacher_model else None,
        trusted_teacher_required=bool(cfg.trusted_teacher_required),
        epochs=max(1, int(cfg.feature_epochs)),
        batch=int(cfg.batch),
        imgsz=int(cfg.imgsz),
        lr=float(cfg.recovery_lr if "clean" in phase_name.lower() or "recovery" in phase_name.lower() else cfg.lr),
        weight_decay=float(cfg.weight_decay),
        num_workers=int(cfg.num_workers),
        device=str(cfg.device) if cfg.device is not None else None,
        max_train_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None,
        max_val_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None,
        amp=bool(cfg.amp),
        max_hook_layers=int(cfg.max_hook_layers),
        prototype_max_batches=int(cfg.prototype_max_batches),
        target_class_ids=[int(x) for x in target_ids],
        **weights,
    )
    report = run_strong_detox_training(fcfg)
    return Path(report.get("best_model") or report.get("final_model"))


def _run_clean_recovery_finetune(
    model: str | Path,
    data_yaml: str | Path,
    out_project: str | Path,
    cfg: HybridPurifyConfig,
    epochs: int | None = None,
) -> Path:
    train_counterfactual_finetune(
        base_model=model,
        data_yaml=data_yaml,
        output_project=out_project,
        name="clean_recovery",
        imgsz=cfg.imgsz,
        epochs=max(1, int(epochs if epochs is not None else cfg.recovery_epochs)),
        batch=cfg.batch,
        device=cfg.device,
        lr0=cfg.recovery_lr,
        weight_decay=cfg.weight_decay,
        mosaic=0.6,
        mixup=0.08,
        copy_paste=0.05,
        erasing=0.20,
        hsv_h=0.03,
        hsv_s=0.45,
        hsv_v=0.35,
        label_smoothing=0.03,
        close_mosaic=1,
    )
    return find_ultralytics_weight(out_project, "clean_recovery", prefer="best")


def _passes(best: Mapping[str, Any], cfg: HybridPurifyConfig) -> bool:
    if float(best.get("external_max_asr", 0.0)) > float(cfg.max_allowed_external_asr):
        return False
    if float(best.get("internal_max_asr", 0.0)) > float(cfg.max_allowed_internal_asr):
        return False
    drop = best.get("map_drop")
    if drop is not None and float(drop) > float(cfg.max_map_drop):
        return False
    if cfg.min_map50_95 is not None:
        metrics = best.get("clean_metrics") or {}
        if "map50_95" in metrics and float(metrics.get("map50_95") or 0.0) < float(cfg.min_map50_95):
            return False
    return True


def run_hybrid_purify_detox_yolo(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    output_dir: str | Path,
    teacher_model: str | Path | None = None,
    cfg: HybridPurifyConfig | None = None,
) -> Dict[str, Any]:
    """Run the strongest generic detox pipeline currently implemented.

    The selection metric is external-hard-suite first. Internal synthetic ASR is
    retained only as a secondary regularizer because the user's experiments show
    internal ASR can be self-consistent while external hard suites fail.
    """
    cfg = cfg or HybridPurifyConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names_from_yaml_or_mapping(data_yaml)
    target_ids = resolve_class_ids(names, target_classes)
    if not target_ids:
        raise ValueError("Hybrid PURIFY requires explicit target_classes")

    replay_roots = list(cfg.external_replay_roots or cfg.external_eval_roots or [])
    eval_roots = list(cfg.external_eval_roots or cfg.external_replay_roots or [])
    replay_datasets = discover_external_attack_datasets(replay_roots)
    external_eval_cfg = ExternalHardSuiteConfig(
        roots=tuple(eval_roots),
        imgsz=cfg.imgsz,
        max_images_per_attack=cfg.external_eval_max_images_per_attack,
        replay_max_images_per_attack=cfg.external_replay_max_images_per_attack,
        seed=cfg.seed,
    )

    manifest: Dict[str, Any] = {
        "algorithm": "Hybrid-PURIFY-OD",
        "description": "External hard-suite + PGBD-style prototypes + I-BAU + NAD/distillation + phase-separated detox",
        "input_model": str(model_path),
        "teacher_model": str(teacher_model) if teacher_model else None,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml": str(data_yaml),
        "target_classes": [str(x) for x in target_classes],
        "target_class_ids": target_ids,
        "config": {**asdict(cfg), "attack_specs": [asdict(a) for a in cfg.attack_specs]},
        "external_replay_datasets": [asdict(ds) for ds in replay_datasets],
        "cycles": [],
        "best": None,
        "status": "running",
        "warnings": [],
    }
    if not teacher_model:
        manifest["warnings"].append("teacher_model not provided; feature distillation uses a frozen copy of the suspicious model, which is weaker.")
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)

    current_model = Path(model_path)
    baseline_cfg = ASRClosedLoopConfig(
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        seed=cfg.seed,
        external_eval_roots=tuple(eval_roots),
        external_replay_roots=tuple(replay_roots),
        external_eval_max_images_per_attack=cfg.external_eval_max_images_per_attack,
        external_replay_max_images_per_attack=cfg.external_replay_max_images_per_attack,
        attack_specs=cfg.attack_specs,
        include_internal_asr=cfg.include_internal_asr,
        eval_max_images=cfg.eval_max_images,
        max_allowed_external_asr=cfg.max_allowed_external_asr,
        max_allowed_internal_asr=cfg.max_allowed_internal_asr,
        max_map_drop=cfg.max_map_drop,
    )
    before_eval = _evaluate_all(
        current_model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        data_yaml=data_yaml,
        target_classes=target_classes,
        cfg=baseline_cfg,
        external_eval_cfg=external_eval_cfg,
        output_dir=output_dir,
        tag="00_before",
    )
    clean_before = before_eval.get("clean_metrics")
    manifest["before_eval"] = {
        "external_max_asr": _max_asr(before_eval.get("external")),
        "internal_max_asr": _max_asr(before_eval.get("internal")),
        "clean_metrics": clean_before,
    }
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)

    best_item: Dict[str, Any] | None = None
    hard_scores = _combined_scores(before_eval)
    external_rows: Sequence[Mapping[str, Any]] | None = (before_eval.get("external") or {}).get("rows")
    baseline_external_asr = _max_asr(before_eval.get("external"))
    baseline_internal_asr = _max_asr(before_eval.get("internal"))
    baseline_external_mean = _mean_asr(before_eval.get("external"))
    baseline_internal_mean = _mean_asr(before_eval.get("internal"))
    best_item = {
        "cycle": 0,
        "model": str(current_model),
        "external_max_asr": baseline_external_asr,
        "external_mean_asr": baseline_external_mean,
        "internal_max_asr": baseline_internal_asr,
        "internal_mean_asr": baseline_internal_mean,
        "clean_metrics": clean_before,
        "map_drop": 0.0,
        "selection_score": _selection_score(baseline_external_asr, baseline_internal_asr, 0.0, baseline_cfg, external_mean_asr=baseline_external_mean),
        "external_json": before_eval.get("external_json"),
        "internal_json": before_eval.get("internal_json"),
        "passes": _passes(
            {
                "external_max_asr": baseline_external_asr,
                "internal_max_asr": baseline_internal_asr,
                "map_drop": 0.0,
                "clean_metrics": clean_before,
            },
            cfg,
        ),
        "cycle_info": {"phase": "baseline"},
    }
    manifest["best"] = best_item
    accepted_model = Path(best_item["model"])
    accepted_hard_scores = dict(hard_scores)
    accepted_external_rows = external_rows
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)

    for cycle in range(1, int(cfg.cycles) + 1):
        current_model = accepted_model
        closed_cfg = ASRClosedLoopConfig(
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            seed=cfg.seed + cycle,
            cycles=1,
            max_allowed_external_asr=cfg.max_allowed_external_asr,
            max_allowed_internal_asr=cfg.max_allowed_internal_asr,
            max_map_drop=cfg.max_map_drop,
            val_fraction=cfg.val_fraction,
            max_images=cfg.max_images,
            eval_max_images=cfg.eval_max_images,
            external_eval_roots=tuple(eval_roots),
            external_replay_roots=tuple(replay_roots),
            external_eval_max_images_per_attack=cfg.external_eval_max_images_per_attack,
            external_replay_max_images_per_attack=cfg.external_replay_max_images_per_attack,
            base_clean_repeat=cfg.base_clean_repeat,
            recovery_clean_repeat=cfg.recovery_clean_repeat,
            base_attack_repeat=cfg.base_attack_repeat,
            max_attack_repeat=cfg.max_attack_repeat,
            adaptive_boost=cfg.adaptive_boost,
            active_asr_threshold=cfg.active_asr_threshold,
            top_k_attacks_per_cycle=cfg.top_k_attacks_per_cycle,
            phase_epochs=cfg.phase_epochs,
            recovery_epochs=cfg.recovery_epochs,
            lr0=cfg.lr,
            recovery_lr0=cfg.recovery_lr,
            weight_decay=cfg.weight_decay,
            attack_specs=cfg.attack_specs,
            include_internal_asr=cfg.include_internal_asr,
            use_external_replay=cfg.use_external_replay,
        )
        phases = _build_phase_plan(cfg.attack_specs, accepted_hard_scores, closed_cfg)
        cycle_info: Dict[str, Any] = {"cycle": cycle, "phases": [], "hard_scores_in": dict(accepted_hard_scores)}

        for pi, phase in enumerate(phases, 1):
            phase_yaml = _build_phase_dataset(
                phase,
                cycle=cycle,
                output_dir=output_dir,
                images_dir=images_dir,
                labels_dir=labels_dir,
                names=names,
                target_ids=target_ids,
                cfg=closed_cfg,
                replay_datasets=replay_datasets,
                failure_rows=accepted_external_rows,
            )
            phase_dir = output_dir / f"02_cycle_{cycle:02d}_phase_{pi:02d}_{phase.name}"
            if cfg.run_feature_purifier:
                current_model = _run_feature_purifier_phase(
                    model=current_model,
                    teacher_model=teacher_model,
                    data_yaml=phase_yaml,
                    out_dir=phase_dir / "feature_purify",
                    target_ids=target_ids,
                    phase_name=phase.name,
                    cfg=cfg,
                )
            if cfg.run_clean_recovery_finetune and ("recovery" in phase.name or "clean_anchor" in phase.name):
                current_model = _run_clean_recovery_finetune(
                    model=current_model,
                    data_yaml=phase_yaml,
                    out_project=phase_dir / "ultralytics_recovery",
                    cfg=cfg,
                    epochs=phase.epochs,
                )
            cycle_info["phases"].append({"phase": asdict(phase), "data_yaml": str(phase_yaml), "model_after": str(current_model)})
            write_json(output_dir / "hybrid_purify_manifest.json", manifest)

        evals = _evaluate_all(
            current_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            data_yaml=data_yaml,
            target_classes=target_classes,
            cfg=closed_cfg,
            external_eval_cfg=external_eval_cfg,
            output_dir=output_dir,
            tag=f"cycle_{cycle:02d}",
        )
        external_asr = _max_asr(evals.get("external"))
        internal_asr = _max_asr(evals.get("internal"))
        mean_external = _mean_asr(evals.get("external"))
        mean_internal = _mean_asr(evals.get("internal"))
        map_drop = _map_drop(clean_before, evals.get("clean_metrics"))
        score = _selection_score(external_asr, internal_asr, map_drop, closed_cfg, external_mean_asr=mean_external)
        item = {
            "cycle": cycle,
            "model": str(current_model),
            "external_max_asr": external_asr,
            "external_mean_asr": mean_external,
            "internal_max_asr": internal_asr,
            "internal_mean_asr": mean_internal,
            "clean_metrics": evals.get("clean_metrics"),
            "map_drop": map_drop,
            "selection_score": score,
            "external_json": evals.get("external_json"),
            "internal_json": evals.get("internal_json"),
            "passes": _passes({"external_max_asr": external_asr, "internal_max_asr": internal_asr, "map_drop": map_drop, "clean_metrics": evals.get("clean_metrics")}, cfg),
            "cycle_info": cycle_info,
        }
        manifest["cycles"].append(item)
        improved = item["selection_score"] < float(best_item["selection_score"]) - float(closed_cfg.min_selection_improvement)
        if improved:
            best_item = item
            manifest["best"] = item
            accepted_model = Path(item["model"])
            accepted_hard_scores = _combined_scores(evals)
            accepted_external_rows = (evals.get("external") or {}).get("rows") or accepted_external_rows
            item["rolled_back"] = False
        else:
            item["rolled_back"] = True
            item["rollback_to"] = str(accepted_model)
            item["rollback_reason"] = "no_selection_improvement"
            current_model = accepted_model
        write_json(output_dir / "hybrid_purify_manifest.json", manifest)
        if item["passes"] and cfg.stop_on_pass:
            manifest["status"] = "passed_early"
            break

    if best_item is None:
        manifest["status"] = "failed_no_checkpoint"
    else:
        manifest["final_model"] = best_item["model"]
        manifest["status"] = "passed" if best_item["passes"] else "failed_external_asr_or_map"
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)
    return manifest
