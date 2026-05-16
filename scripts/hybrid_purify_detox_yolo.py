#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, split_known_keys, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hybrid-PURIFY-OD: external hard-suite + feature-level YOLO backdoor detox")
    p.add_argument("--config", default=None, help="YAML config. Values under hybrid_purify_detox: are accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None)
    p.add_argument("--teacher-model", default=None, help="Trusted clean teacher checkpoint. Strongly recommended.")
    p.add_argument("--images", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--external-eval-roots", nargs="*", default=None)
    p.add_argument("--external-replay-roots", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--phase-epochs", type=int, default=None)
    p.add_argument("--feature-epochs", type=int, default=None)
    p.add_argument("--recovery-epochs", type=int, default=None)
    p.add_argument("--max-allowed-external-asr", type=float, default=None)
    p.add_argument("--max-allowed-internal-asr", type=float, default=None)
    p.add_argument("--max-map-drop", type=float, default=None)
    p.add_argument("--selection-max-map-drop", type=float, default=None)
    p.add_argument("--min-map50-95", type=float, default=None)
    p.add_argument("--external-eval-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-floor-per-attack", type=int, default=None,
                   help="Patch B: guarantee at least N real replay samples per attack even when failure_only is True.")
    p.add_argument("--head-only-blacklist", default=None,
                   help="Patch G: path to head_only_blacklist.json. Images listed are excluded from OGA negative training.")
    p.add_argument("--min-passing-eval-n-per-attack", type=int, default=None,
                   help="Fix F1: refuse to declare 'passed' / early-exit when any attack's eval sample size is below N. "
                        "Prevents small-sample early-exits like v3 (60-img eval max_asr=5%% -> passed, but full-300 max was 9.46%%).")
    p.add_argument("--output-distill-scale", type=float, default=None,
                   help="Patch D: multiplier applied on top of phase-wise lambda_output_distill. "
                        "Set to 0.0 to disable output distillation when the teacher's final decisions are not trusted.")
    p.add_argument("--feature-distill-scale", type=float, default=None,
                   help="Patch D: multiplier applied on top of phase-wise lambda_feature_distill. Default 1.0.")
    p.add_argument("--external-failure-replay-repeat", type=int, default=None)
    p.add_argument("--external-oda-full-image-extra-repeat", type=int, default=None, help="Extra full-image repeats for ODA failure replay, preserving trigger/context")
    p.add_argument("--external-oda-focus-crops", action="store_true", default=None, help="Add target-centered crops for ODA failure replay samples")
    p.add_argument("--external-oda-focus-crop-repeat", type=int, default=None)
    p.add_argument("--external-oda-focus-crop-context", type=float, default=None)
    p.add_argument("--external-oda-focus-crop-min-size", type=int, default=None)
    p.add_argument(
        "--external-oda-success-mode",
        choices=["localized_any_recalled", "class_presence", "strict_all_recalled"],
        default=None,
        help="ODA ASR definition used for external hard-suite evaluation.",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--eval-max-images", type=int, default=None)
    p.add_argument("--no-feature-purifier", action="store_true", default=None)
    p.add_argument("--allow-self-teacher-feature-purifier", action="store_true", default=None)
    p.add_argument("--no-phase-finetune", action="store_true", default=None)
    p.add_argument("--no-clean-recovery-finetune", action="store_true", default=None)
    p.add_argument("--no-evaluate-each-phase", action="store_true", default=None, help="Disable phase-level external ASR checkpoint selection")
    p.add_argument("--no-rollback-bad-phase", action="store_true", default=None, help="Continue from a phase even if it worsens an external attack")
    p.add_argument("--rollback-unimproved-phase", action="store_true", default=None, help="Rollback phase candidates that do not improve external selection score")
    p.add_argument("--no-external-failure-replay", action="store_true", default=None, help="Use generic external replay instead of current success=true failures only")
    p.add_argument("--no-external-select-phase-checkpoints", action="store_true", default=None, help="Only evaluate the primary feature-purifier checkpoint per phase")
    p.add_argument("--aggressive-mode", action="store_true", default=None, help="Train harder on top external failures, then rely on rollback gates")
    p.add_argument("--aggressive-feature-epochs", type=int, default=None)
    p.add_argument("--aggressive-phase-epochs", type=int, default=None)
    p.add_argument("--aggressive-lr-multiplier", type=float, default=None)
    p.add_argument("--aggressive-adv-steps", type=int, default=None)
    p.add_argument("--aggressive-failure-replay-repeat", type=int, default=None)
    p.add_argument("--aggressive-lambda-oda-recall", type=float, default=None)
    p.add_argument("--aggressive-lambda-oda-matched", type=float, default=None)
    p.add_argument("--aggressive-lambda-pgbd-paired", type=float, default=None)
    p.add_argument("--aggressive-lambda-oga-negative", type=float, default=None)
    p.add_argument("--oda-recall-min-conf", type=float, default=None)
    p.add_argument("--oda-recall-iou-threshold", type=float, default=None)
    p.add_argument("--oda-recall-center-radius", type=float, default=None)
    p.add_argument("--oda-recall-topk", type=int, default=None)
    p.add_argument("--oda-recall-loss-scale", type=float, default=None)
    p.add_argument("--oda-matched-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-teacher-score-weight", type=float, default=None)
    p.add_argument("--oda-matched-teacher-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-min-score", type=float, default=None)
    p.add_argument("--oda-matched-best-score-weight", type=float, default=None)
    p.add_argument("--oda-matched-best-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-localized-margin", type=float, default=None)
    p.add_argument("--oda-matched-localized-margin-weight", type=float, default=None)
    p.add_argument("--pgbd-view-mode", default=None)
    p.add_argument("--pgbd-negative-margin", type=float, default=None)
    p.add_argument("--trusted-teacher-required", action="store_true", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no-pre-prune", action="store_true", default=None, help="Disable RNP-lite pre-prune candidate")
    p.add_argument("--no-stop-on-pass", action="store_true", default=None, help="Do not early-exit even if an intermediate candidate passes; run all cycles/phases.")
    p.add_argument("--pre-prune-top-k", type=int, default=None)
    p.add_argument("--pre-prune-strength", type=float, default=None)
    p.add_argument("--rnp-unlearn-steps", type=int, default=None)
    p.add_argument("--rnp-max-images", type=int, default=None)
    p.add_argument("--allow-attack-worse", action="store_true", default=None, help="Allow candidates that worsen a single external attack; not recommended")
    p.add_argument("--max-single-attack-asr-worsen", type=float, default=None)
    p.add_argument("--external-mean-asr-weight", type=float, default=None)
    p.add_argument("--min-external-asr-improvement", type=float, default=None)
    p.add_argument("--min-external-mean-improvement", type=float, default=None)

    # Lagrangian multi-attack controller (opt-in, main-line algorithmic extension).
    p.add_argument(
        "--use-lagrangian-controller",
        action="store_true",
        default=None,
        help="Enable adaptive per-attack Lagrangian lambda updates across cycles. "
        "When off, phase lambdas remain static (backward compatible).",
    )
    p.add_argument("--lagrangian-lambda-lr", type=float, default=None)
    p.add_argument("--lagrangian-lambda-min", type=float, default=None)
    p.add_argument("--lagrangian-lambda-max", type=float, default=None)
    p.add_argument("--lagrangian-decay", type=float, default=None)
    p.add_argument("--lagrangian-base-scale", type=float, default=None)
    p.add_argument("--lagrangian-max-scale", type=float, default=None)
    p.add_argument("--lagrangian-min-scale", type=float, default=None)
    return p.parse_args()


def _resolved(args: argparse.Namespace) -> dict:
    from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig

    defaults = {
        "model": None,
        "teacher_model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "out": "runs/hybrid_purify_detox",
    }
    defaults.update(HybridPurifyConfig().__dict__)
    raw = load_yaml_config(args.config, section="hybrid_purify_detox")
    cli = namespace_overrides(args, exclude={"config"})
    bool_map = {
        "no_feature_purifier": ("run_feature_purifier", False),
        "allow_self_teacher_feature_purifier": ("allow_self_teacher_feature_purifier", True),
        "no_phase_finetune": ("run_phase_finetune", False),
        "no_clean_recovery_finetune": ("run_clean_recovery_finetune", False),
        "no_evaluate_each_phase": ("evaluate_each_phase", False),
        "no_rollback_bad_phase": ("rollback_bad_phase", False),
        "rollback_unimproved_phase": ("rollback_unimproved_phase", True),
        "no_external_failure_replay": ("external_failure_replay", False),
        "external_oda_focus_crops": ("external_oda_focus_crops", True),
        "no_external_select_phase_checkpoints": ("external_select_phase_checkpoints", False),
        "aggressive_mode": ("aggressive_mode", True),
        "no_pre_prune": ("run_pre_prune", False),
        "no_stop_on_pass": ("stop_on_pass", False),
        "allow_attack_worse": ("require_no_attack_worse", False),
    }
    norm = {}
    for k, v in cli.items():
        if k in bool_map:
            if v:
                nk, nv = bool_map[k]
                norm[nk] = nv
        elif k == "data_yaml":
            norm["data_yaml"] = v
        else:
            norm[k] = v
    return deep_merge(defaults, deep_merge(raw, norm))


def main() -> None:
    args = parse_args()
    from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig, run_hybrid_purify_detox_yolo
    from model_security_gate.detox.asr_aware_dataset import load_attacks_from_config

    r = _resolved(args)
    missing = [k for k in ["model", "images", "labels", "data_yaml", "target_classes"] if not r.get(k)]
    if missing:
        raise SystemExit("Missing required config/CLI values: " + ", ".join(missing))
    cfg_keys = set(HybridPurifyConfig.__dataclass_fields__.keys())
    cfg_data, extra = split_known_keys(r, cfg_keys)
    if "attacks" in r:
        cfg_data["attack_specs"] = load_attacks_from_config(r.get("attacks"))
    # Patch G: load blacklist if provided.
    if cfg_data.get("head_only_blacklist") and not cfg_data.get("blacklist_stems"):
        from model_security_gate.detox.asr_aware_dataset import load_blacklist
        cfg_data["blacklist_stems"] = tuple(load_blacklist(cfg_data["head_only_blacklist"]))
    for list_key in ["external_eval_roots", "external_replay_roots"]:
        if cfg_data.get(list_key) is None:
            cfg_data[list_key] = ()
        elif isinstance(cfg_data.get(list_key), list):
            cfg_data[list_key] = tuple(cfg_data[list_key])
    cfg = HybridPurifyConfig(**cfg_data)
    out_dir = Path(str(r.get("out") or "runs/hybrid_purify_detox"))
    write_resolved_config(out_dir / "resolved_config.json", r)
    manifest = run_hybrid_purify_detox_yolo(
        model_path=r["model"],
        teacher_model=r.get("teacher_model"),
        images_dir=r["images"],
        labels_dir=r["labels"],
        data_yaml=r["data_yaml"],
        target_classes=r["target_classes"],
        output_dir=out_dir,
        cfg=cfg,
    )
    print(f"[DONE] status: {manifest.get('status')}")
    print(f"[DONE] final model: {manifest.get('final_model')}")
    print(f"[DONE] manifest: {out_dir / 'hybrid_purify_manifest.json'}")


if __name__ == "__main__":
    main()
