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
    p.add_argument("--min-map50-95", type=float, default=None)
    p.add_argument("--external-eval-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-max-images-per-attack", type=int, default=None)
    p.add_argument(
        "--external-oda-success-mode",
        choices=["localized_any_recalled", "class_presence", "strict_all_recalled"],
        default=None,
        help="ODA ASR definition used for external hard-suite evaluation.",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--eval-max-images", type=int, default=None)
    p.add_argument("--no-feature-purifier", action="store_true", default=None)
    p.add_argument("--no-clean-recovery-finetune", action="store_true", default=None)
    p.add_argument("--trusted-teacher-required", action="store_true", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no-pre-prune", action="store_true", default=None, help="Disable RNP-lite pre-prune candidate")
    p.add_argument("--pre-prune-top-k", type=int, default=None)
    p.add_argument("--pre-prune-strength", type=float, default=None)
    p.add_argument("--rnp-unlearn-steps", type=int, default=None)
    p.add_argument("--rnp-max-images", type=int, default=None)
    p.add_argument("--allow-attack-worse", action="store_true", default=None, help="Allow candidates that worsen a single external attack; not recommended")
    p.add_argument("--max-single-attack-asr-worsen", type=float, default=None)
    p.add_argument("--external-mean-asr-weight", type=float, default=None)
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
        "no_clean_recovery_finetune": ("run_clean_recovery_finetune", False),
        "no_pre_prune": ("run_pre_prune", False),
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
