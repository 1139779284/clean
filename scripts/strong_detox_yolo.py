#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, split_known_keys, write_resolved_config


def parse_args():
    p = argparse.ArgumentParser(description="Run the full strong detox pipeline on top of the existing Model Security Gate project")
    p.add_argument("--config", default=None, help="YAML config. Values under `strong_detox:` are also accepted. CLI args override YAML.")
    p.add_argument("--model", default=None, help="Suspicious YOLO .pt model")
    p.add_argument("--images", default=None, help="Clean/shadow image directory in YOLO layout")
    p.add_argument("--labels", default=None, help="YOLO labels directory for --images. Optional for --label-mode pseudo/feature_only")
    p.add_argument("--data-yaml", default=None, help="YOLO data.yaml with class names")
    p.add_argument("--target-classes", nargs="*", default=None, help="Critical class names or ids, e.g. helmet. Omit to scan/detox all classes")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--trusted-base-model", default=None, help="Trusted official/pretrained checkpoint used to train a clean teacher")
    p.add_argument("--teacher-model", default=None, help="Already trained clean teacher .pt. Overrides --trusted-base-model training")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-scan-images", type=int, default=None)
    p.add_argument("--max-feature-images", type=int, default=None, help="0 means all generated CF train images")
    p.add_argument("--cf-finetune-epochs", type=int, default=None)
    p.add_argument("--teacher-epochs", type=int, default=None)
    p.add_argument("--nad-epochs", type=int, default=None)
    p.add_argument("--ibau-epochs", type=int, default=None)
    p.add_argument("--prototype-epochs", type=int, default=None)
    p.add_argument("--prune-top-k", type=int, default=None)
    p.add_argument("--prune-top-ks", type=int, nargs="*", default=None)
    p.add_argument("--no-anp-scan", action="store_true", default=None)
    p.add_argument("--no-progressive-prune", action="store_true", default=None)
    p.add_argument("--skip-teacher-train", action="store_true", default=None)
    p.add_argument("--skip-prune", action="store_true", default=None)
    p.add_argument("--skip-cf-finetune", action="store_true", default=None)
    p.add_argument("--skip-nad", action="store_true", default=None)
    p.add_argument("--skip-ibau", action="store_true", default=None)
    p.add_argument("--skip-prototype", action="store_true", default=None)
    p.add_argument("--label-mode", choices=["auto", "supervised", "pseudo", "feature_only"], default=None, help="supervised uses true labels; pseudo builds a detox set from teacher/self pseudo labels; feature_only avoids supervised CF training")
    p.add_argument("--pseudo-source", choices=["agreement", "teacher", "suspicious"], default=None, help="Pseudo-label source when --label-mode pseudo and a teacher is available")
    p.add_argument("--pseudo-conf", type=float, default=None, help="Teacher confidence threshold for pseudo labels")
    p.add_argument("--pseudo-min-suspicious-conf", type=float, default=None, help="Suspicious-model confidence threshold used for agreement")
    p.add_argument("--pseudo-max-conf-gap", type=float, default=None, help="Reject teacher/suspicious pseudo matches with larger confidence gap")
    p.add_argument("--pseudo-agreement-iou", type=float, default=None, help="Teacher/suspicious agreement IoU for pseudo labels")
    p.add_argument("--no-pseudo-reject-if-teacher-empty", action="store_true", default=None, help="Do not reject images when teacher returns no boxes")
    p.add_argument("--no-save-rejected-pseudo", action="store_true", default=None, help="Do not copy rejected pseudo-label images")
    p.add_argument("--no-rerun-security-gate", action="store_true", default=None, help="Skip automatic post-detox security_gate verification")
    p.add_argument("--verify-occlusion", action="store_true", default=None, help="Run occlusion scan during automatic post-detox verification")
    p.add_argument("--verify-channel", action="store_true", default=None, help="Run channel scan during automatic post-detox verification")
    p.add_argument("--verify-max-images", type=int, default=None)
    p.add_argument("--fail-on-verify-error", action="store_true", default=None, help="Exit non-zero if automatic post-detox verification fails")
    return p.parse_args()


def _load_resolved(args):
    from model_security_gate.detox.strong_pipeline import StrongDetoxConfig

    raw = load_yaml_config(args.config, section="strong_detox")
    defaults = {
        "out": "runs/strong_detox",
        "model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "trusted_base_model": None,
        "teacher_model": None,
    }
    defaults.update(StrongDetoxConfig().__dict__)
    cli = namespace_overrides(args, exclude={"config"})
    # Convert legacy negative flags into positive config keys.
    bool_map = {
        "no_anp_scan": ("run_anp_scan", False),
        "no_progressive_prune": ("run_progressive_prune", False),
        "no_pseudo_reject_if_teacher_empty": ("pseudo_reject_if_teacher_empty", False),
        "no_save_rejected_pseudo": ("pseudo_save_rejected_samples", False),
        "no_rerun_security_gate": ("rerun_security_gate", False),
        "verify_occlusion": ("run_occlusion_verify", True),
        "verify_channel": ("run_channel_verify", True),
    }
    normalized_cli = {}
    for key, value in cli.items():
        if key in bool_map:
            if value:
                new_key, new_value = bool_map[key]
                normalized_cli[new_key] = new_value
        elif key == "data_yaml":
            normalized_cli["data_yaml"] = value
        else:
            normalized_cli[key] = value
    resolved = deep_merge(defaults, deep_merge(raw, normalized_cli))
    return resolved


def main() -> None:
    args = parse_args()
    from model_security_gate.detox.strong_pipeline import StrongDetoxConfig, run_strong_detox_pipeline

    resolved = _load_resolved(args)
    required = ["model", "images", "data_yaml"]
    missing = [k for k in required if not resolved.get(k)]
    if missing:
        raise SystemExit(f"Missing required config/CLI values: {', '.join(missing)}")

    cfg_keys = set(StrongDetoxConfig.__dataclass_fields__.keys())
    cfg_data, _extra = split_known_keys(resolved, cfg_keys)
    if "prune_top_ks" in cfg_data and isinstance(cfg_data["prune_top_ks"], list):
        cfg_data["prune_top_ks"] = tuple(cfg_data["prune_top_ks"])
    cfg = StrongDetoxConfig(**cfg_data)
    out_dir = Path(str(resolved.get("out") or "runs/strong_detox"))
    write_resolved_config(out_dir / "resolved_config.json", resolved)

    manifest = run_strong_detox_pipeline(
        suspicious_model=resolved["model"],
        images_dir=resolved["images"],
        labels_dir=resolved.get("labels"),
        data_yaml=resolved["data_yaml"],
        target_classes=resolved.get("target_classes"),
        output_dir=out_dir,
        trusted_base_model=resolved.get("trusted_base_model"),
        teacher_model=resolved.get("teacher_model"),
        cfg=cfg,
    )
    print(f"[DONE] final model: {manifest.get('final_model')}")
    print(f"[DONE] manifest: {out_dir / 'strong_detox_manifest.json'}")
    print(f"[DONE] resolved config: {out_dir / 'resolved_config.json'}")


if __name__ == "__main__":
    main()
