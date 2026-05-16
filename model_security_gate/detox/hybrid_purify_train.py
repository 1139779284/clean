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
from model_security_gate.detox.multi_attack_constraints import (
    AttackConstraint,
    MultiAttackLagrangianController,
    default_t0_constraints,
)
from model_security_gate.detox.strong_train import StrongDetoxConfig as FeatureStrongDetoxConfig
from model_security_gate.detox.strong_train import run_strong_detox_training
from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.detox.common import find_ultralytics_weight
from model_security_gate.detox.rnp import RNPConfig, apply_rnp_soft_suppression, score_rnp_channels_for_yolo
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
    selection_max_map_drop: float | None = None
    min_map50_95: float | None = None
    val_fraction: float = 0.15
    max_images: int = 0
    eval_max_images: int = 0
    external_eval_roots: Sequence[str] = field(default_factory=tuple)
    external_replay_roots: Sequence[str] = field(default_factory=tuple)
    external_eval_max_images_per_attack: int = 0
    external_replay_max_images_per_attack: int = 250
    external_oda_success_mode: str = "localized_any_recalled"
    external_failure_replay: bool = True
    external_failure_replay_repeat: int = 4
    external_replay_floor_per_attack: int = 80
    external_oda_full_image_extra_repeat: int = 0
    external_oda_focus_crops: bool = False
    external_oda_focus_crop_repeat: int = 2
    external_oda_focus_crop_context: float = 3.0
    external_oda_focus_crop_min_size: int = 160

    # Patch G: path to head_only_blacklist.json or direct list of stems.
    head_only_blacklist: str | None = None
    blacklist_stems: Sequence[str] = field(default_factory=tuple)

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
    num_workers: int = 0
    amp: bool = False
    max_hook_layers: int = 6
    prototype_max_batches: int = 40

    # Pipeline switches.
    use_external_replay: bool = True
    include_internal_asr: bool = True
    stop_on_pass: bool = True
    # Fix F1: minimum number of per-attack samples required before the
    # outer loop is allowed to declare "passed" and early-exit.  When each
    # attack's eval sample size is below this, a point-estimate max ASR
    # under the threshold is not enough evidence because the Wilson 95%
    # upper bound can span well above ``max_allowed_external_asr``.
    # Empirically, 60 images/attack yields +-6% CI width which is too wide
    # for a 10% threshold; 120 images/attack cuts that to +-4% which is
    # safe.  Default stays at 0 (off) for backward compatibility; the
    # ablation configs opt in at 120.
    min_passing_eval_n_per_attack: int = 0
    # Patch D: global scale applied on top of the phase-wise
    # ``lambda_output_distill`` values.  Use 0.0 to fully disable output
    # distillation when the teacher is imperfect (e.g. a 3-epoch clean
    # teacher that may wrongly call a sinusoidal blend pattern "helmet").
    # Feature distillation and NAD attention are unaffected; the teacher
    # still provides trustworthy intermediate features while its final
    # decisions are ignored.  Default 1.0 preserves the original behaviour.
    output_distill_scale: float = 1.0
    # Patch D symmetric knob.  Keeps the feature-layer distillation signal
    # fully on (1.0) even when output distill is off.
    feature_distill_scale: float = 1.0
    run_feature_purifier: bool = True
    allow_self_teacher_feature_purifier: bool = False
    run_phase_finetune: bool = True
    run_clean_recovery_finetune: bool = True
    trusted_teacher_required: bool = False
    evaluate_each_phase: bool = True
    rollback_bad_phase: bool = True
    rollback_unimproved_phase: bool = False
    external_select_phase_checkpoints: bool = True

    # Aggressive-but-rollback mode: train harder on current failures, but only
    # accept checkpoints that pass the external ASR / mAP / per-attack gates.
    aggressive_mode: bool = False
    aggressive_feature_epochs: int = 4
    aggressive_phase_epochs: int = 2
    aggressive_lr_multiplier: float = 2.0
    aggressive_adv_steps: int = 4
    aggressive_failure_replay_repeat: int = 8
    aggressive_top_k_attacks_per_cycle: int = 2
    defer_clean_recovery_in_aggressive: bool = True
    aggressive_lambda_oda_recall: float = 2.0
    oda_recall_min_conf: float = 0.45
    oda_recall_iou_threshold: float = 0.05
    oda_recall_center_radius: float = 1.50
    oda_recall_topk: int = 24
    oda_recall_loss_scale: float = 1.0
    aggressive_lambda_oda_matched: float = 1.0
    aggressive_lambda_pgbd_paired: float = 0.70
    aggressive_lambda_oga_negative: float = 0.80
    oda_matched_box_weight: float = 0.25
    oda_matched_teacher_score_weight: float = 0.25
    oda_matched_teacher_box_weight: float = 0.10
    oda_matched_min_score: float = 0.50
    oda_matched_best_score_weight: float = 0.90
    oda_matched_best_box_weight: float = 0.35
    oda_matched_localized_margin: float = 0.10
    oda_matched_localized_margin_weight: float = 0.25
    pgbd_view_mode: str = "mixed"
    pgbd_negative_margin: float = 0.25

    # Conservative RNP-lite soft-pruning. This is not a hard requirement for
    # acceptance; it is evaluated as a candidate and rolled back if external ASR
    # or clean mAP worsens. Defaults are intentionally soft.
    run_pre_prune: bool = True
    pre_prune_top_k: int = 32
    pre_prune_strength: float = 0.72
    rnp_unlearn_steps: int = 40
    rnp_max_images: int = 96
    rnp_min_score_to_prune: float = 0.03

    # Pareto safety constraints. A candidate that improves mean score but makes
    # one critical attack worse is rejected unless explicitly disabled.
    require_no_attack_worse: bool = True
    max_single_attack_asr_worsen: float = 0.02
    external_mean_asr_weight: float = 0.35
    internal_asr_weight: float = 0.05
    worse_attack_penalty: float = 2.5
    oda_worse_penalty: float = 3.0
    min_selection_improvement: float = 0.005
    min_external_asr_improvement: float = 1e-6
    min_external_mean_improvement: float = 0.01

    # Lagrangian multi-attack controller (opt-in).
    #
    # When enabled, the Hybrid-PURIFY cycle loop feeds every cycle's external
    # per-attack ASR into a MultiAttackLagrangianController and multiplies each
    # phase's loss weights by the current per-attack lambda.  This keeps the
    # training constrained-optimization aware of per-attack violation levels
    # instead of relying on a static ``aggressive_mode`` boolean.  Default is
    # off so every current run is bit-for-bit identical unless the user opts
    # in via config.
    use_lagrangian_controller: bool = False
    lagrangian_lambda_lr: float = 0.25
    lagrangian_lambda_min: float = 0.0
    lagrangian_lambda_max: float = 6.0
    lagrangian_decay: float = 0.98
    lagrangian_base_scale: float = 1.0
    lagrangian_max_scale: float = 4.0
    lagrangian_min_scale: float = 0.5

    attack_specs: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())



def _asr_matrix(result: Mapping[str, Any] | None) -> Dict[str, float]:
    try:
        matrix = ((result or {}).get("summary") or {}).get("asr_matrix") or {}
        return {str(k): float(v) for k, v in dict(matrix).items()}
    except Exception:
        return {}


def compare_asr_matrices(
    baseline: Mapping[str, float] | None,
    candidate: Mapping[str, float] | None,
    max_worsen: float = 0.02,
) -> Dict[str, Any]:
    """Compare per-attack ASR and flag regressions.

    This is the guard that prevents the exact failure the user observed where a
    balanced run improved clean mAP but made badnet_oda worse. The comparison is
    key-wise and tolerant of missing keys; missing candidate keys are ignored,
    because some external suites may be absent in small runs.
    """
    base = {str(k): float(v) for k, v in dict(baseline or {}).items()}
    cand = {str(k): float(v) for k, v in dict(candidate or {}).items()}
    rows: List[Dict[str, Any]] = []
    worse: List[Dict[str, Any]] = []
    for key, after in sorted(cand.items()):
        before = base.get(key)
        if before is None:
            continue
        delta = float(after) - float(before)
        row = {"attack": key, "before": float(before), "after": float(after), "delta": delta, "worse": delta > float(max_worsen)}
        rows.append(row)
        if row["worse"]:
            worse.append(row)
    return {"rows": rows, "worse": worse, "n_worse": len(worse), "max_worsen": float(max_worsen)}


def _hybrid_selection_score(
    external_asr: float,
    internal_asr: float,
    external_mean_asr: float,
    map_drop: float | None,
    worse_compare: Mapping[str, Any] | None,
    cfg: HybridPurifyConfig,
) -> float:
    # External max ASR dominates, external mean ASR prevents hiding broad failure
    # behind one improved attack, and per-attack worsening is heavily penalized.
    score = (
        1.35 * float(external_asr)
        + float(cfg.internal_asr_weight) * float(internal_asr)
        + float(cfg.external_mean_asr_weight) * float(external_mean_asr)
    )
    selection_max_drop = cfg.selection_max_map_drop if cfg.selection_max_map_drop is not None else cfg.max_map_drop
    if map_drop is not None and float(map_drop) > float(selection_max_drop):
        score += 10.0 * (float(map_drop) - float(selection_max_drop))
    worse_rows = list((worse_compare or {}).get("worse") or [])
    if worse_rows:
        score += float(cfg.worse_attack_penalty) * len(worse_rows)
        if any("oda" in str(r.get("attack", "")).lower() for r in worse_rows):
            score += float(cfg.oda_worse_penalty)
    return float(score)

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
            "lambda_oda_recall": 0.0,
            "lambda_oda_matched": 0.0,
            "lambda_oga_negative": 0.75,
            "lambda_pgbd_paired": 0.45,
        }
    if "oda" in low:
        return {
            "lambda_task": 0.85,            # was 1.45; task loss dominated recall before
            "lambda_adv": 0.25,
            "lambda_output_distill": 0.45,
            "lambda_feature_distill": 0.55, # slight boost
            "lambda_nad": 0.50,
            "lambda_attention": 0.40,
            "lambda_prototype": 0.55,
            "lambda_proto_suppress": 0.05,  # keep low to avoid ODA suppression
            "lambda_oda_recall": 1.80,      # was 1.0, now 2x
            "lambda_oda_matched": 1.20,     # was 0.75
            "lambda_oga_negative": 0.0,
            "lambda_pgbd_paired": 0.55,
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
            "lambda_oda_recall": 0.25,
            "lambda_oda_matched": 0.35,
            "lambda_oga_negative": 0.35,
            "lambda_pgbd_paired": 0.80,
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
            "lambda_oda_recall": 0.35,
            "lambda_oda_matched": 0.45,
            "lambda_oga_negative": 0.15,
            "lambda_pgbd_paired": 0.80,
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
        "lambda_oda_recall": 0.0,
        "lambda_oda_matched": 0.0,
        "lambda_oga_negative": 0.0,
        "lambda_pgbd_paired": 0.0,
    }


# ---------------------------------------------------------------------------
# Lagrangian multi-attack controller wiring
# ---------------------------------------------------------------------------
#
# The controller is intentionally constrained to a small, auditable set of
# phase-level loss multipliers.  The training loop in ``strong_train.py``
# already implements every loss term; the controller's job is only to decide
# how strongly each per-attack family should be weighted in the next cycle
# given the observed external ASR violation.
#
# Mapping attack -> phase:
#   badnet_oga,   blend_oga  -> oga bucket (lambda_oga_negative, proto_suppress)
#   badnet_oda,   wanet_oda  -> oda bucket (lambda_oda_recall, oda_matched)
#   wanet_oga,    wanet_*    -> wanet bucket (lambda_pgbd_paired, pgbd proto)
#   semantic_*               -> semantic bucket (lambda_pgbd_paired, proto_suppress)
#   clean_map_drop           -> recovery bucket (lambda_output_distill)
#
# Phase lambda scale is ``base_scale + lambda * per_unit`` clamped to
# ``[lagrangian_min_scale, lagrangian_max_scale]``.


_ATTACK_TO_BUCKET: Dict[str, str] = {
    "badnet_oga": "oga",
    "badnet_oga_corner": "oga",
    "blend_oga": "oga",
    "semantic_cleanlabel": "semantic",
    "semantic_green_cleanlabel": "semantic",
    "semantic_fp_max_conf": "semantic",
    "badnet_oda": "oda",
    "wanet_oda": "oda",
    "wanet_oga": "wanet",
    "wanet_oga_corner": "wanet",
    "clean_map_drop": "clean",
}

_BUCKET_LAMBDA_KEYS: Dict[str, tuple[str, ...]] = {
    "oga": ("lambda_oga_negative", "lambda_proto_suppress", "lambda_pgbd_paired"),
    "oda": ("lambda_oda_recall", "lambda_oda_matched", "lambda_attention"),
    "wanet": ("lambda_pgbd_paired", "lambda_feature_distill", "lambda_nad"),
    "semantic": ("lambda_pgbd_paired", "lambda_proto_suppress", "lambda_feature_distill"),
    "clean": ("lambda_output_distill", "lambda_feature_distill", "lambda_nad"),
}


def _bucket_for_attack(name: str) -> str | None:
    low = str(name).strip().lower()
    if low in _ATTACK_TO_BUCKET:
        return _ATTACK_TO_BUCKET[low]
    for key, bucket in _ATTACK_TO_BUCKET.items():
        if low.startswith(key) or key.startswith(low):
            return bucket
    return None


def _build_lagrangian_controller(cfg: HybridPurifyConfig) -> MultiAttackLagrangianController:
    """Construct the controller from ``default_t0_constraints`` + cfg knobs."""

    base = default_t0_constraints()
    constraints = [
        AttackConstraint(
            name=c.name,
            metric=c.metric,
            max_value=c.max_value,
            baseline_value=c.baseline_value,
            no_worse_epsilon=c.no_worse_epsilon,
            weight=c.weight,
            direction=c.direction,
            hard=c.hard,
        )
        for c in base
    ]
    return MultiAttackLagrangianController(
        constraints=constraints,
        lambda_lr=float(cfg.lagrangian_lambda_lr),
        lambda_min=float(cfg.lagrangian_lambda_min),
        lambda_max=float(cfg.lagrangian_lambda_max),
        decay=float(cfg.lagrangian_decay),
    )


def _bucket_scales_from_lambdas(lambdas: Mapping[str, float], cfg: HybridPurifyConfig) -> Dict[str, float]:
    """Aggregate per-attack lambdas into per-bucket scale multipliers.

    Each bucket's scale is ``base_scale + mean(bucket_lambdas)/lambda_max``
    times the base range, clamped to ``[min_scale, max_scale]``.
    """

    lmax = max(1e-6, float(cfg.lagrangian_lambda_max))
    buckets: Dict[str, list[float]] = {}
    for name, value in (lambdas or {}).items():
        bucket = _bucket_for_attack(str(name))
        if bucket is None:
            continue
        buckets.setdefault(bucket, []).append(float(value))
    scales: Dict[str, float] = {}
    for bucket, values in buckets.items():
        norm = (sum(values) / len(values)) / lmax if values else 0.0
        raw = float(cfg.lagrangian_base_scale) + norm * (
            float(cfg.lagrangian_max_scale) - float(cfg.lagrangian_base_scale)
        )
        scales[bucket] = max(
            float(cfg.lagrangian_min_scale),
            min(float(cfg.lagrangian_max_scale), raw),
        )
    return scales


def _apply_lagrangian_weights(
    base: Mapping[str, float],
    phase_name: str,
    lambdas: Mapping[str, float] | None,
    cfg: HybridPurifyConfig,
) -> Dict[str, float]:
    """Scale the phase's lambda_* entries by the controller's per-bucket scale.

    Only lambda keys listed in the active bucket are scaled.  Other keys are
    passed through unchanged so unrelated losses (e.g. lambda_task) keep the
    weights designed for each phase.
    """

    out = {k: float(v) for k, v in base.items()}
    if not lambdas:
        return out
    scales = _bucket_scales_from_lambdas(lambdas, cfg)
    if not scales:
        return out
    low = str(phase_name).lower()
    active: list[str] = []
    if "oga" in low:
        active.append("oga")
    if "oda" in low:
        active.append("oda")
    if "wanet" in low or "warp" in low:
        active.append("wanet")
    if "semantic" in low:
        active.append("semantic")
    if "clean" in low or "recovery" in low:
        active.append("clean")
    if not active:
        return out
    for bucket in active:
        scale = scales.get(bucket)
        if scale is None:
            continue
        for key in _BUCKET_LAMBDA_KEYS.get(bucket, ()):
            if key in out:
                out[key] = float(out[key]) * float(scale)
    return out


def _normalize_metric_keys(metrics: Mapping[str, float]) -> Dict[str, float]:
    """Normalize external ASR matrix keys for controller consumption.

    External hard-suite reports typically name keys as ``suite::attack``.
    The controller's default constraints are keyed by attack name only.  We
    strip any ``suite::`` prefix and merge duplicates by taking the max.
    Some local suites include provenance suffixes in the attack name
    (for example ``badnet_oga_mask_bd_v2_visible``); those should still
    drive the canonical ``badnet_oga`` controller constraint.
    """

    out: Dict[str, float] = {}
    aliases = (
        "badnet_oda",
        "badnet_oga",
        "blend_oga",
        "wanet_oga",
        "wanet_oda",
        "semantic_cleanlabel",
        "semantic_fp_max_conf",
    )
    for key, value in (metrics or {}).items():
        name = str(key).split("::")[-1].strip().lower()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        out[name] = max(out.get(name, 0.0), numeric)
        for alias in aliases:
            if alias in name:
                out[alias] = max(out.get(alias, 0.0), numeric)
    return out


def _run_feature_purifier_phase(
    model: str | Path,
    teacher_model: str | Path | None,
    data_yaml: str | Path,
    out_dir: str | Path,
    target_ids: Sequence[int],
    phase_name: str,
    cfg: HybridPurifyConfig,
    lagrangian_lambdas: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    weights = _phase_feature_weights(phase_name)
    phase_low = phase_name.lower()
    aggressive = bool(cfg.aggressive_mode and "clean" not in phase_low and "recovery" not in phase_low)
    if aggressive:
        weights = dict(weights)
        weights["lambda_adv"] = max(float(weights.get("lambda_adv", 0.0)), 0.65)
        weights["lambda_output_distill"] = min(float(weights.get("lambda_output_distill", 0.0)), 0.35)
        weights["lambda_feature_distill"] = max(float(weights.get("lambda_feature_distill", 0.0)), 0.55)
        weights["lambda_nad"] = max(float(weights.get("lambda_nad", 0.0)), 0.60)
        if "oga" in phase_low or "semantic" in phase_low:
            weights["lambda_proto_suppress"] = max(float(weights.get("lambda_proto_suppress", 0.0)), 1.20)
        if "oda" in phase_low:
            weights["lambda_task"] = max(float(weights.get("lambda_task", 0.0)), 1.80)
            weights["lambda_attention"] = max(float(weights.get("lambda_attention", 0.0)), 0.55)
            weights["lambda_prototype"] = max(float(weights.get("lambda_prototype", 0.0)), 1.10)
            weights["lambda_proto_suppress"] = min(float(weights.get("lambda_proto_suppress", 0.0)), 0.05)
            weights["lambda_oda_recall"] = max(
                float(weights.get("lambda_oda_recall", 0.0)),
                float(cfg.aggressive_lambda_oda_recall),
            )
            weights["lambda_oda_matched"] = max(
                float(weights.get("lambda_oda_matched", 0.0)),
                float(cfg.aggressive_lambda_oda_matched),
            )
            weights["lambda_pgbd_paired"] = max(
                float(weights.get("lambda_pgbd_paired", 0.0)),
                float(cfg.aggressive_lambda_pgbd_paired),
            )
        if "oga" in phase_low or "semantic" in phase_low or "wanet" in phase_low:
            weights["lambda_pgbd_paired"] = max(
                float(weights.get("lambda_pgbd_paired", 0.0)),
                float(cfg.aggressive_lambda_pgbd_paired),
            )
        if "oga" in phase_low:
            weights["lambda_oga_negative"] = max(
                float(weights.get("lambda_oga_negative", 0.0)),
                float(cfg.aggressive_lambda_oga_negative),
            )
    epochs = max(1, int(cfg.aggressive_feature_epochs if aggressive else cfg.feature_epochs))
    lr = float(cfg.recovery_lr if "clean" in phase_low or "recovery" in phase_low else cfg.lr)
    if aggressive:
        lr *= float(cfg.aggressive_lr_multiplier)

    # Apply adaptive Lagrangian per-bucket scaling on top of the static and
    # aggressive weights.  This is a no-op when ``lagrangian_lambdas`` is None
    # or empty, which keeps backward compatibility for existing callers.
    pre_lagrangian_weights = dict(weights)
    weights = _apply_lagrangian_weights(weights, phase_name, lagrangian_lambdas, cfg)

    # Patch D: apply teacher-side scales.  When the teacher's decisions are
    # not trusted (e.g. a lightly-finetuned clean teacher may hallucinate
    # helmet on blend-pattern backgrounds), setting ``output_distill_scale=0``
    # disables output-level distillation while keeping feature-level signals.
    if "lambda_output_distill" in weights:
        weights["lambda_output_distill"] = float(weights["lambda_output_distill"]) * float(cfg.output_distill_scale)
    if "lambda_feature_distill" in weights:
        weights["lambda_feature_distill"] = float(weights["lambda_feature_distill"]) * float(cfg.feature_distill_scale)

    # Patch E: pick a PGBD paired-view mode that matches the current phase
    # instead of using a single global "mixed" mode for every phase.  The
    # default is a phase-inferred mode; if the caller explicitly set a
    # non-default ``pgbd_view_mode`` we honor it unchanged.
    from model_security_gate.detox.pgbd_od import infer_pgbd_mode_from_phase

    base_pgbd_mode = str(cfg.pgbd_view_mode or "mixed")
    if base_pgbd_mode.lower() in {"auto", "mixed"}:
        pgbd_view_mode = infer_pgbd_mode_from_phase(phase_name, default=base_pgbd_mode)
    else:
        pgbd_view_mode = base_pgbd_mode

    fcfg = FeatureStrongDetoxConfig(
        model=str(model),
        data_yaml=str(data_yaml),
        out_dir=str(out_dir),
        teacher_model=str(teacher_model) if teacher_model else None,
        trusted_teacher_required=bool(cfg.trusted_teacher_required),
        epochs=epochs,
        batch=int(cfg.batch),
        imgsz=int(cfg.imgsz),
        lr=lr,
        weight_decay=float(cfg.weight_decay),
        num_workers=int(cfg.num_workers),
        device=str(cfg.device) if cfg.device is not None else None,
        max_train_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None,
        max_val_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None,
        amp=bool(cfg.amp),
        max_hook_layers=int(cfg.max_hook_layers),
        prototype_max_batches=int(cfg.prototype_max_batches),
        target_class_ids=[int(x) for x in target_ids],
        adv_steps=int(cfg.aggressive_adv_steps if aggressive else 2),
        oda_recall_min_conf=float(cfg.oda_recall_min_conf),
        oda_recall_iou_threshold=float(cfg.oda_recall_iou_threshold),
        oda_recall_center_radius=float(cfg.oda_recall_center_radius),
        oda_recall_topk=int(cfg.oda_recall_topk),
        oda_recall_loss_scale=float(cfg.oda_recall_loss_scale),
        oda_matched_box_weight=float(cfg.oda_matched_box_weight),
        oda_matched_teacher_score_weight=float(cfg.oda_matched_teacher_score_weight),
        oda_matched_teacher_box_weight=float(cfg.oda_matched_teacher_box_weight),
        oda_matched_min_score=float(cfg.oda_matched_min_score),
        oda_matched_best_score_weight=float(cfg.oda_matched_best_score_weight),
        oda_matched_best_box_weight=float(cfg.oda_matched_best_box_weight),
        oda_matched_localized_margin=float(cfg.oda_matched_localized_margin),
        oda_matched_localized_margin_weight=float(cfg.oda_matched_localized_margin_weight),
        pgbd_view_mode=str(pgbd_view_mode),
        pgbd_negative_margin=float(cfg.pgbd_negative_margin),
        save_every=1 if cfg.external_select_phase_checkpoints else max(1, epochs),
        **weights,
    )
    report = run_strong_detox_training(fcfg)
    candidate_paths: List[Path] = []
    for key in ["best_model", "final_model"]:
        if report.get(key):
            candidate_paths.append(Path(str(report[key])))
    if cfg.external_select_phase_checkpoints:
        candidate_paths.extend(sorted(Path(out_dir).glob("epoch_*.pt")))
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidate_paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    primary = Path(report.get("final_model") or report.get("best_model") or unique[-1])
    return {
        "primary_model": str(primary),
        "candidates": [str(path) for path in unique],
        "report": report,
        "aggressive": aggressive,
        "weights": dict(weights),
        "weights_pre_lagrangian": pre_lagrangian_weights,
        "lagrangian_lambdas": {str(k): float(v) for k, v in (lagrangian_lambdas or {}).items()},
        "pgbd_view_mode": str(pgbd_view_mode),
    }


def _run_clean_recovery_finetune(
    model: str | Path,
    data_yaml: str | Path,
    out_project: str | Path,
    cfg: HybridPurifyConfig,
    epochs: int | None = None,
) -> Path:
    # Ultralytics interprets a relative ``project`` path under its own
    # ``runs/detect/`` root, which would cause ``find_ultralytics_weight`` to
    # look in the wrong place.  Force an absolute path so downstream lookups
    # agree on a single directory.
    out_project_abs = Path(out_project).resolve()
    out_project_abs.mkdir(parents=True, exist_ok=True)
    train_counterfactual_finetune(
        base_model=model,
        data_yaml=data_yaml,
        output_project=out_project_abs,
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
    return find_ultralytics_weight(out_project_abs, "clean_recovery", prefer="best")


def _run_phase_finetune(
    model: str | Path,
    data_yaml: str | Path,
    out_project: str | Path,
    cfg: HybridPurifyConfig,
    phase_name: str,
    epochs: int | None = None,
    lagrangian_lambdas: Mapping[str, float] | None = None,
) -> List[Path]:
    """Run a supervised YOLO fine-tune on the current hardening phase dataset.

    This is the safe fallback when no trusted clean teacher is available. The
    phase dataset already contains failure-only external replay samples, and the
    outer loop still decides by external ASR / clean mAP rather than train loss.

    When ``lagrangian_lambdas`` is provided, the phase's base learning rate is
    scaled by the square root of the per-bucket scale from the Lagrangian
    controller (clamped to ``[lagrangian_min_scale, lagrangian_max_scale]``).
    Using sqrt keeps the lr change moderate relative to the bucket scale used
    for loss weights, which matters for Ultralytics' built-in optimiser stability.
    """
    # Same absolute-path fix as _run_clean_recovery_finetune: Ultralytics
    # otherwise interprets a relative ``project`` under ``runs/detect/`` and
    # ``find_ultralytics_weight`` ends up searching the wrong tree.
    out_project_abs = Path(out_project).resolve()
    out_project_abs.mkdir(parents=True, exist_ok=True)
    lr0 = float(cfg.lr) * float(cfg.aggressive_lr_multiplier if cfg.aggressive_mode else 1.0)
    if lagrangian_lambdas:
        scales = _bucket_scales_from_lambdas(lagrangian_lambdas, cfg)
        low = phase_name.lower()
        active_scales = []
        for bucket in ("oga", "oda", "wanet", "semantic", "clean"):
            if bucket in low or (bucket == "wanet" and "warp" in low):
                if bucket in scales:
                    active_scales.append(scales[bucket])
        if active_scales:
            # sqrt to avoid dramatic lr shifts; still preserves the ordering.
            mean_scale = sum(active_scales) / len(active_scales)
            lr0 = lr0 * (mean_scale ** 0.5)
    train_counterfactual_finetune(
        base_model=model,
        data_yaml=data_yaml,
        output_project=out_project_abs,
        name="phase_finetune",
        imgsz=cfg.imgsz,
        epochs=max(1, int(epochs if epochs is not None else cfg.phase_epochs)),
        batch=cfg.batch,
        device=cfg.device,
        lr0=lr0,
        weight_decay=cfg.weight_decay,
        mosaic=0.25 if "oda" in phase_name else 0.45,
        mixup=0.03 if "oda" in phase_name else 0.08,
        copy_paste=0.02,
        erasing=0.08 if "oda" in phase_name else 0.18,
        hsv_h=0.02,
        hsv_s=0.35,
        hsv_v=0.30,
        label_smoothing=0.01 if "oda" in phase_name else 0.03,
        close_mosaic=1,
        workers=cfg.num_workers,
    )
    weights_dir = Path(out_project_abs) / "phase_finetune" / "weights"
    candidates: List[Path] = []
    for prefer in ("best", "last"):
        try:
            path = find_ultralytics_weight(out_project_abs, "phase_finetune", prefer=prefer)
        except FileNotFoundError:
            continue
        if path.exists() and path not in candidates:
            candidates.append(path)
    if not candidates and weights_dir.exists():
        candidates.extend(sorted(weights_dir.glob("*.pt")))
    return candidates


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
    if bool(cfg.require_no_attack_worse) and int((best.get("asr_compare_to_baseline") or {}).get("n_worse", 0) or 0) > 0:
        return False
    # Fix F1: refuse to declare "passed" when the underlying external
    # evaluation is too small for its Wilson 95% upper bound to stay below
    # ``max_allowed_external_asr``.  Concretely: if any tracked attack has
    # ``n_rows < min_passing_eval_n_per_attack``, the small-sample CI could
    # easily span above the threshold even if the point estimate is under,
    # so early-exit would be unsafe.  See docs/V3_ABLATION_RESULTS_2026-05-11.md
    # for the concrete failure this guard prevents (OGA-only early-exit
    # on 60-image eval caused WaNet regression on the full 300-image eval).
    min_n = int(getattr(cfg, "min_passing_eval_n_per_attack", 0) or 0)
    if min_n > 0:
        counts = best.get("external_attack_counts") or {}
        if counts:
            try:
                min_observed = min(int(v) for v in counts.values() if v is not None)
            except (TypeError, ValueError):
                min_observed = None
            if min_observed is not None and min_observed < min_n:
                return False
    return True


def _candidate_block_reasons(item: Mapping[str, Any], cfg: HybridPurifyConfig) -> List[str]:
    reasons: List[str] = []
    if bool(cfg.require_no_attack_worse) and int((item.get("asr_compare_to_baseline") or {}).get("n_worse", 0) or 0) > 0:
        reasons.append("attack_worse_than_baseline")
    drop = item.get("map_drop")
    selection_max_drop = cfg.selection_max_map_drop if cfg.selection_max_map_drop is not None else cfg.max_map_drop
    if drop is not None and float(drop) > float(selection_max_drop):
        reasons.append("map_drop_exceeds_threshold")
    return reasons


def _candidate_improved(item: Mapping[str, Any], best_item: Mapping[str, Any], cfg: HybridPurifyConfig) -> bool:
    """Decide whether a candidate should replace the current best.

    External max ASR is the hard safety signal. A candidate that increases the
    current best max ASR must not replace it just because mean ASR, mAP, or the
    aggregate score looks better. This keeps last-mile runs from selecting a
    broad-but-worse checkpoint after a phase already found a lower max-ASR one.
    """
    try:
        best_max = float(best_item.get("external_max_asr", 0.0))
        item_max = float(item.get("external_max_asr", 0.0))
        best_mean = float(best_item.get("external_mean_asr", 0.0))
        item_mean = float(item.get("external_mean_asr", 0.0))
    except (TypeError, ValueError):
        return False

    max_eps = max(float(cfg.min_external_asr_improvement), 1e-9)
    if item_max > best_max + max_eps:
        return False
    if best_max - item_max > max_eps:
        return True

    same_max = abs(best_max - item_max) <= max_eps
    if not same_max:
        return False
    if best_mean - item_mean >= float(cfg.min_external_mean_improvement):
        return True

    score_delta = float(best_item["selection_score"]) - float(item["selection_score"])
    return bool(score_delta > float(cfg.min_selection_improvement))


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
    """Run Hybrid-PURIFY-OD v2.

    This version is deliberately conservative: every candidate is evaluated on
    the external hard suite, compared per-attack against the original baseline,
    and rolled back if it worsens any critical attack or harms clean mAP. The
    pipeline can therefore safely try stronger RNP/feature-level detox steps
    without letting a bad candidate contaminate later cycles.
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
        oda_success_mode=cfg.external_oda_success_mode,
    )

    manifest: Dict[str, Any] = {
        "algorithm": "Hybrid-PURIFY-OD-v2",
        "description": "External hard-suite selection + RNP-lite candidate + PGBD/I-BAU/NAD phase purifier + rollback",
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
    feature_purifier_enabled = bool(cfg.run_feature_purifier)
    if not teacher_model:
        if bool(cfg.run_feature_purifier) and not bool(cfg.allow_self_teacher_feature_purifier):
            feature_purifier_enabled = False
            manifest["warnings"].append(
                "teacher_model not provided; feature purifier disabled to avoid self-teacher backdoor distillation. "
                "Using failure-only phase fine-tune fallback."
            )
        else:
            manifest["warnings"].append("teacher_model not provided; feature distillation uses a frozen copy of the suspicious model, which is weaker.")
    if not eval_roots:
        manifest["warnings"].append("No external_eval_roots provided; Hybrid-PURIFY will rely on internal ASR and is not a reliable external-hard-suite solution.")
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
    baseline_external_asr = _max_asr(before_eval.get("external"))
    baseline_internal_asr = _max_asr(before_eval.get("internal"))
    baseline_external_mean = _mean_asr(before_eval.get("external"))
    baseline_internal_mean = _mean_asr(before_eval.get("internal"))
    baseline_external_matrix = _asr_matrix(before_eval.get("external"))
    manifest["before_eval"] = {
        "external_max_asr": baseline_external_asr,
        "external_mean_asr": baseline_external_mean,
        "internal_max_asr": baseline_internal_asr,
        "internal_mean_asr": baseline_internal_mean,
        "clean_metrics": clean_before,
        "external_asr_matrix": baseline_external_matrix,
    }

    best_item: Dict[str, Any] = {
        "cycle": 0,
        "model": str(current_model),
        "external_max_asr": baseline_external_asr,
        "external_mean_asr": baseline_external_mean,
        "internal_max_asr": baseline_internal_asr,
        "internal_mean_asr": baseline_internal_mean,
        "clean_metrics": clean_before,
        "map_drop": 0.0,
        "selection_score": _hybrid_selection_score(baseline_external_asr, baseline_internal_asr, baseline_external_mean, 0.0, None, cfg),
        "external_json": before_eval.get("external_json"),
        "internal_json": before_eval.get("internal_json"),
        "asr_compare_to_baseline": {"rows": [], "worse": [], "n_worse": 0},
        "passes": _passes({"external_max_asr": baseline_external_asr, "internal_max_asr": baseline_internal_asr, "map_drop": 0.0, "clean_metrics": clean_before}, cfg),
        "cycle_info": {"phase": "baseline"},
    }
    manifest["best"] = best_item
    accepted_model = Path(best_item["model"])
    accepted_hard_scores = _combined_scores(before_eval)
    accepted_external_rows = (before_eval.get("external") or {}).get("rows")
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)

    def evaluate_candidate(candidate_model: Path, tag: str, cycle_info: Mapping[str, Any]) -> Dict[str, Any]:
        evals = _evaluate_all(
            candidate_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            data_yaml=data_yaml,
            target_classes=target_classes,
            cfg=baseline_cfg,
            external_eval_cfg=external_eval_cfg,
            output_dir=output_dir,
            tag=tag,
        )
        ext = _max_asr(evals.get("external"))
        inte = _max_asr(evals.get("internal"))
        mean_ext = _mean_asr(evals.get("external"))
        mean_int = _mean_asr(evals.get("internal"))
        drop = _map_drop(clean_before, evals.get("clean_metrics"))
        asr_compare = compare_asr_matrices(baseline_external_matrix, _asr_matrix(evals.get("external")), cfg.max_single_attack_asr_worsen)
        # Fix F1: extract per-attack sample counts so _passes can refuse
        # early-exit when the eval is too small for reliable CI.
        external_attack_counts: Dict[str, int] = {}
        top_attacks = ((evals.get("external") or {}).get("summary") or {}).get("top_attacks") or []
        for ta in top_attacks:
            if isinstance(ta, Mapping) and "attack" in ta and "n" in ta:
                external_attack_counts[str(ta["attack"])] = int(ta["n"])
        return {
            "model": str(candidate_model),
            "external_max_asr": ext,
            "external_mean_asr": mean_ext,
            "internal_max_asr": inte,
            "internal_mean_asr": mean_int,
            "clean_metrics": evals.get("clean_metrics"),
            "map_drop": drop,
            "selection_score": _hybrid_selection_score(ext, inte, mean_ext, drop, asr_compare, cfg),
            "external_json": evals.get("external_json"),
            "internal_json": evals.get("internal_json"),
            "asr_compare_to_baseline": asr_compare,
            "external_attack_counts": external_attack_counts,
            "passes": _passes({"external_max_asr": ext, "internal_max_asr": inte, "map_drop": drop, "clean_metrics": evals.get("clean_metrics"), "asr_compare_to_baseline": asr_compare, "external_attack_counts": external_attack_counts}, cfg),
            "cycle_info": dict(cycle_info),
            "_evals": evals,
        }

    def consider_candidate(
        item: Dict[str, Any],
        evals: Mapping[str, Any],
        *,
        rollback_on_blocked: bool,
        rollback_on_no_improvement: bool,
    ) -> tuple[Dict[str, Any], bool, bool]:
        nonlocal best_item, accepted_model, accepted_hard_scores, accepted_external_rows

        block_reasons = _candidate_block_reasons(item, cfg)
        blocked = bool(block_reasons)
        improved = _candidate_improved(item, best_item, cfg)
        public_item = {k: v for k, v in item.items() if k != "_evals"}
        public_item["blocked"] = blocked
        public_item["block_reasons"] = block_reasons
        public_item["improved"] = improved

        if improved and not blocked:
            public_item["accepted_as_best"] = True
            public_item["rolled_back"] = False
            best_item = public_item
            manifest["best"] = best_item
            accepted_model = Path(public_item["model"])
            accepted_hard_scores = _combined_scores(evals)
            accepted_external_rows = (evals.get("external") or {}).get("rows") or accepted_external_rows
            return public_item, True, False

        public_item["accepted_as_best"] = False
        if blocked:
            public_item["rollback_reason"] = "+".join(block_reasons)
            public_item["rolled_back"] = bool(rollback_on_blocked)
        else:
            public_item["rollback_reason"] = "no_selection_improvement"
            public_item["rolled_back"] = bool(rollback_on_no_improvement)
        if public_item["rolled_back"]:
            public_item["rollback_to"] = str(accepted_model)
        return public_item, False, bool(public_item["rolled_back"])

    # RNP-lite candidate before gradient-heavy purification.
    if cfg.run_pre_prune and int(cfg.pre_prune_top_k) > 0:
        rnp_dir = output_dir / "00_rnp_candidate"
        rnp_dir.mkdir(parents=True, exist_ok=True)
        try:
            rnp_cfg = RNPConfig(
                imgsz=cfg.imgsz,
                batch=max(1, min(int(cfg.batch), 4)),
                device=cfg.device,
                max_images=int(cfg.rnp_max_images),
                unlearn_steps=int(cfg.rnp_unlearn_steps),
                soft_suppression_strength=float(cfg.pre_prune_strength),
                min_score_to_prune=float(cfg.rnp_min_score_to_prune),
            )
            score_csv, _summary = score_rnp_channels_for_yolo(current_model, data_yaml, rnp_dir / "rnp_scores.csv", rnp_cfg)
            rnp_model = apply_rnp_soft_suppression(
                current_model,
                score_csv,
                rnp_dir / "rnp_soft_suppressed.pt",
                top_k=int(cfg.pre_prune_top_k),
                strength=float(cfg.pre_prune_strength),
                min_score=float(cfg.rnp_min_score_to_prune),
                device=cfg.device,
            )
            rnp_item = evaluate_candidate(Path(rnp_model), tag="00_rnp_candidate", cycle_info={"phase": "rnp_candidate", "score_csv": str(score_csv)})
            rnp_item["cycle"] = 0
            manifest["rnp_candidate"] = {k: v for k, v in rnp_item.items() if k != "_evals"}
            improved = _candidate_improved(rnp_item, best_item, cfg)
            block_reasons = _candidate_block_reasons(rnp_item, cfg)
            blocked = bool(block_reasons)
            if improved and not blocked:
                best_item = {k: v for k, v in rnp_item.items() if k != "_evals"}
                manifest["best"] = best_item
                accepted_model = Path(best_item["model"])
                accepted_hard_scores = _combined_scores(rnp_item["_evals"])
                accepted_external_rows = (rnp_item["_evals"].get("external") or {}).get("rows") or accepted_external_rows
            else:
                manifest["rnp_candidate"]["rolled_back"] = True
                manifest["rnp_candidate"]["block_reasons"] = block_reasons
                manifest["rnp_candidate"]["rollback_reason"] = "+".join(block_reasons) if blocked else "no_selection_improvement"
        except Exception as exc:  # noqa: BLE001
            manifest["warnings"].append(f"RNP pre-prune candidate failed and was skipped: {exc}")
        write_json(output_dir / "hybrid_purify_manifest.json", manifest)

    # Lagrangian controller initialization.
    #
    # When ``use_lagrangian_controller`` is True, we seed the controller with
    # the baseline external ASR matrix so the first cycle already uses
    # non-default lambdas for attacks that are currently violated.  When
    # disabled, ``controller`` stays None and the training path is identical
    # to the pre-Lagrangian behaviour.
    controller: MultiAttackLagrangianController | None = None
    if bool(cfg.use_lagrangian_controller):
        controller = _build_lagrangian_controller(cfg)
        seed_metrics = dict(baseline_external_matrix)
        if clean_before is not None:
            try:
                seed_metrics["clean_map_drop"] = 0.0  # baseline has zero drop by definition
            except Exception:
                pass
        controller.update(_normalize_metric_keys(seed_metrics))
        manifest["lagrangian_controller"] = {
            "enabled": True,
            "config": controller.to_dict(),
            "seed_metrics": seed_metrics,
            "trace": [],
        }
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
            external_failure_replay=bool(cfg.external_failure_replay),
            external_failure_replay_repeat=int(
                cfg.aggressive_failure_replay_repeat if cfg.aggressive_mode else cfg.external_failure_replay_repeat
            ),
            external_replay_floor_per_attack=int(cfg.external_replay_floor_per_attack),
            external_oda_full_image_extra_repeat=int(cfg.external_oda_full_image_extra_repeat),
            external_oda_focus_crops=bool(cfg.external_oda_focus_crops),
            external_oda_focus_crop_repeat=int(cfg.external_oda_focus_crop_repeat),
            external_oda_focus_crop_context=float(cfg.external_oda_focus_crop_context),
            external_oda_focus_crop_min_size=int(cfg.external_oda_focus_crop_min_size),
            blacklist_stems=list(cfg.blacklist_stems or []),
            base_clean_repeat=cfg.base_clean_repeat,
            recovery_clean_repeat=cfg.recovery_clean_repeat,
            base_attack_repeat=cfg.base_attack_repeat,
            max_attack_repeat=cfg.max_attack_repeat,
            adaptive_boost=cfg.adaptive_boost,
            active_asr_threshold=cfg.active_asr_threshold,
            top_k_attacks_per_cycle=int(cfg.aggressive_top_k_attacks_per_cycle if cfg.aggressive_mode else cfg.top_k_attacks_per_cycle),
            phase_epochs=int(cfg.aggressive_phase_epochs if cfg.aggressive_mode else cfg.phase_epochs),
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
        stop_after_phase = False

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
            phase_entry: Dict[str, Any] = {"phase": asdict(phase), "data_yaml": str(phase_yaml), "evaluations": []}
            if feature_purifier_enabled:
                current_lambdas = controller.lambdas if controller is not None else None
                feature_result = _run_feature_purifier_phase(
                    model=current_model,
                    teacher_model=teacher_model,
                    data_yaml=phase_yaml,
                    out_dir=phase_dir / "feature_purify",
                    target_ids=target_ids,
                    phase_name=phase.name,
                    cfg=cfg,
                    lagrangian_lambdas=current_lambdas,
                )
                phase_entry["feature_purifier"] = {
                    "primary_model": feature_result.get("primary_model"),
                    "candidates": feature_result.get("candidates", []),
                    "aggressive": feature_result.get("aggressive"),
                    "weights": feature_result.get("weights"),
                    "weights_pre_lagrangian": feature_result.get("weights_pre_lagrangian"),
                    "lagrangian_lambdas": feature_result.get("lagrangian_lambdas"),
                }
                current_model = Path(str(feature_result.get("primary_model")))
                phase_entry["model_after_feature"] = str(current_model)
                if cfg.evaluate_each_phase:
                    accepted_in_phase = False
                    rollback_in_phase = False
                    candidates = [Path(str(p)) for p in feature_result.get("candidates", [])]
                    if not candidates:
                        candidates = [Path(current_model)]
                    for ci, candidate in enumerate(candidates, 1):
                        feature_item = evaluate_candidate(
                            Path(candidate),
                            tag=f"cycle_{cycle:02d}_phase_{pi:02d}_{phase.name}_feature_c{ci:02d}_{candidate.stem}",
                            cycle_info={
                                "cycle": cycle,
                                "phase": asdict(phase),
                                "phase_index": pi,
                                "stage": "feature_purify",
                                "candidate_index": ci,
                                "candidate_name": candidate.name,
                            },
                        )
                        feature_item["cycle"] = cycle
                        feature_item["phase_index"] = pi
                        feature_item["stage"] = "feature_purify"
                        feature_item["candidate_index"] = ci
                        feature_item["candidate_name"] = candidate.name
                        feature_evals = feature_item.pop("_evals")
                        public_feature, accepted, should_rollback = consider_candidate(
                            feature_item,
                            feature_evals,
                            rollback_on_blocked=bool(cfg.rollback_bad_phase),
                            rollback_on_no_improvement=bool(cfg.rollback_unimproved_phase),
                        )
                        phase_entry["evaluations"].append(public_feature)
                        accepted_in_phase = accepted_in_phase or accepted
                        rollback_in_phase = rollback_in_phase or should_rollback
                        if accepted and public_feature.get("passes") and cfg.stop_on_pass:
                            stop_after_phase = True
                    if accepted_in_phase or rollback_in_phase or bool(cfg.rollback_unimproved_phase):
                        current_model = accepted_model
                        phase_entry["model_after_feature_selected"] = str(current_model)
            run_phase_finetune_now = bool(cfg.run_phase_finetune) and any(
                token in phase.name for token in ("oga", "oda", "semantic", "wanet", "hardening")
            )
            if run_phase_finetune_now:
                phase_ft_dir = phase_dir / "ultralytics_phase_finetune"
                phase_candidates = _run_phase_finetune(
                    model=current_model,
                    data_yaml=phase_yaml,
                    out_project=phase_ft_dir,
                    cfg=cfg,
                    phase_name=phase.name,
                    epochs=phase.epochs,
                    lagrangian_lambdas=(controller.lambdas if controller is not None else None),
                )
                phase_entry["phase_finetune"] = {"candidates": [str(p) for p in phase_candidates]}
                if cfg.evaluate_each_phase:
                    accepted_in_phase = False
                    rollback_in_phase = False
                    for ci, candidate in enumerate(phase_candidates, 1):
                        phase_item = evaluate_candidate(
                            Path(candidate),
                            tag=f"cycle_{cycle:02d}_phase_{pi:02d}_{phase.name}_phaseft_c{ci:02d}_{candidate.stem}",
                            cycle_info={
                                "cycle": cycle,
                                "phase": asdict(phase),
                                "phase_index": pi,
                                "stage": "phase_finetune",
                                "candidate_index": ci,
                                "candidate_name": candidate.name,
                            },
                        )
                        phase_item["cycle"] = cycle
                        phase_item["phase_index"] = pi
                        phase_item["stage"] = "phase_finetune"
                        phase_item["candidate_index"] = ci
                        phase_item["candidate_name"] = candidate.name
                        phase_evals = phase_item.pop("_evals")
                        public_phase, accepted, should_rollback = consider_candidate(
                            phase_item,
                            phase_evals,
                            rollback_on_blocked=bool(cfg.rollback_bad_phase),
                            rollback_on_no_improvement=True,
                        )
                        phase_entry["evaluations"].append(public_phase)
                        accepted_in_phase = accepted_in_phase or accepted
                        rollback_in_phase = rollback_in_phase or should_rollback
                        if accepted and public_phase.get("passes") and cfg.stop_on_pass:
                            stop_after_phase = True
                    current_model = accepted_model
                    phase_entry["model_after_phase_finetune_selected"] = str(current_model)
            run_recovery_now = cfg.run_clean_recovery_finetune and ("recovery" in phase.name or "clean_anchor" in phase.name)
            if cfg.aggressive_mode and cfg.defer_clean_recovery_in_aggressive and phase.name == "clean_anchor":
                run_recovery_now = False
                phase_entry["clean_recovery_skipped"] = "deferred_by_aggressive_mode"
            if run_recovery_now:
                current_model = _run_clean_recovery_finetune(
                    model=current_model,
                    data_yaml=phase_yaml,
                    out_project=phase_dir / "ultralytics_recovery",
                    cfg=cfg,
                    epochs=phase.epochs,
                )
                phase_entry["model_after_recovery"] = str(current_model)
                if cfg.evaluate_each_phase:
                    recovery_item = evaluate_candidate(
                        Path(current_model),
                        tag=f"cycle_{cycle:02d}_phase_{pi:02d}_{phase.name}_recovery",
                        cycle_info={
                            "cycle": cycle,
                            "phase": asdict(phase),
                            "phase_index": pi,
                            "stage": "clean_recovery",
                        },
                    )
                    recovery_item["cycle"] = cycle
                    recovery_item["phase_index"] = pi
                    recovery_item["stage"] = "clean_recovery"
                    recovery_evals = recovery_item.pop("_evals")
                    public_recovery, accepted, should_rollback = consider_candidate(
                        recovery_item,
                        recovery_evals,
                        rollback_on_blocked=bool(cfg.rollback_bad_phase),
                        rollback_on_no_improvement=False,
                    )
                    phase_entry["evaluations"].append(public_recovery)
                    if should_rollback:
                        current_model = accepted_model
                        phase_entry["model_after_recovery_rollback"] = str(current_model)
                    if accepted and public_recovery.get("passes") and cfg.stop_on_pass:
                        stop_after_phase = True
            phase_entry["model_after"] = str(current_model)
            cycle_info["phases"].append(phase_entry)
            write_json(output_dir / "hybrid_purify_manifest.json", manifest)
            if stop_after_phase:
                break

        item = evaluate_candidate(Path(current_model), tag=f"cycle_{cycle:02d}", cycle_info=cycle_info)
        item["cycle"] = cycle
        evals = item.pop("_evals")
        public_item, _accepted, should_rollback = consider_candidate(
            item,
            evals,
            rollback_on_blocked=True,
            rollback_on_no_improvement=True,
        )
        if should_rollback:
            current_model = accepted_model
        manifest["cycles"].append(public_item)

        # Feed observed per-attack ASR into the controller so the next cycle's
        # lambda weights reflect current violation levels.
        if controller is not None:
            cycle_matrix = _asr_matrix(evals.get("external"))
            metric_inputs = _normalize_metric_keys(cycle_matrix)
            try:
                metric_inputs["clean_map_drop"] = float(public_item.get("map_drop", 0.0))
            except (TypeError, ValueError):
                pass
            trace_row = controller.update(metric_inputs)
            manifest.setdefault(
                "lagrangian_controller",
                {"enabled": True, "config": controller.to_dict(), "trace": []},
            )
            manifest["lagrangian_controller"]["trace"].append(
                {
                    "cycle": cycle,
                    "metrics": metric_inputs,
                    "lambdas": dict(controller.lambdas),
                    "updates": trace_row,
                }
            )

        write_json(output_dir / "hybrid_purify_manifest.json", manifest)
        if best_item.get("passes") and cfg.stop_on_pass:
            manifest["status"] = "passed_early"
            break

    manifest["final_model"] = str(accepted_model)
    manifest["status"] = "passed" if best_item.get("passes") else "failed_external_asr_or_map"
    write_json(output_dir / "hybrid_purify_manifest.json", manifest)
    return manifest
