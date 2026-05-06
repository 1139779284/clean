#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.asr_aware_dataset import load_attacks_from_config
from model_security_gate.detox.asr_closed_loop_train import ASRClosedLoopConfig, run_asr_closed_loop_detox_yolo
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, split_known_keys, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run external-hard-suite closed-loop ASR detox for YOLO")
    p.add_argument("--config", default=None, help="YAML config. Values under `asr_closed_loop_detox:` are also accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--external-eval-roots", nargs="*", default=None, help="Benchmark roots used for checkpoint selection, e.g. poison_benchmark_cuda_large poison_benchmark_cuda_tuned")
    p.add_argument("--external-replay-roots", nargs="*", default=None, help="Hard-suite roots replayed into phase datasets. Defaults to external eval roots.")
    p.add_argument("--external-eval-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-max-images-per-attack", type=int, default=None)
    p.add_argument(
        "--external-oda-success-mode",
        choices=["localized_any_recalled", "class_presence", "strict_all_recalled"],
        default=None,
        help="ODA ASR definition used for external hard-suite evaluation.",
    )
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--phase-epochs", type=int, default=None)
    p.add_argument("--recovery-epochs", type=int, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--recovery-lr0", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-allowed-external-asr", type=float, default=None)
    p.add_argument("--max-allowed-internal-asr", type=float, default=None)
    p.add_argument("--max-map-drop", type=float, default=None)
    p.add_argument("--min-map50-95", type=float, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--eval-max-images", type=int, default=None)
    p.add_argument("--base-clean-repeat", type=int, default=None)
    p.add_argument("--recovery-clean-repeat", type=int, default=None)
    p.add_argument("--base-attack-repeat", type=int, default=None)
    p.add_argument("--max-attack-repeat", type=int, default=None)
    p.add_argument("--adaptive-boost", type=float, default=None)
    p.add_argument("--active-asr-threshold", type=float, default=None)
    p.add_argument("--top-k-attacks-per-cycle", type=int, default=None)
    p.add_argument("--no-internal-asr", action="store_true", default=None)
    p.add_argument("--no-external-replay", action="store_true", default=None)
    p.add_argument("--no-stop-on-pass", action="store_true", default=None)
    return p.parse_args()


def _resolved(args: argparse.Namespace) -> dict:
    defaults = {
        "model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "out": "runs/asr_closed_loop_detox",
    }
    defaults.update(ASRClosedLoopConfig().__dict__)
    raw = load_yaml_config(args.config, section="asr_closed_loop_detox")
    cli = namespace_overrides(args, exclude={"config"})
    normalized = {}
    for key, value in cli.items():
        if key == "data_yaml":
            normalized["data_yaml"] = value
        elif key == "no_internal_asr" and value:
            normalized["include_internal_asr"] = False
        elif key == "no_external_replay" and value:
            normalized["use_external_replay"] = False
        elif key == "no_stop_on_pass" and value:
            normalized["stop_on_pass"] = False
        else:
            normalized[key] = value
    return deep_merge(defaults, deep_merge(raw, normalized))


def main() -> None:
    args = parse_args()
    cfg_dict = _resolved(args)
    missing = [k for k in ["model", "images", "labels", "data_yaml", "target_classes"] if not cfg_dict.get(k)]
    if missing:
        raise SystemExit(f"Missing required values: {', '.join(missing)}")
    cfg_fields = set(ASRClosedLoopConfig.__dataclass_fields__.keys())
    cfg_data, _extra = split_known_keys(cfg_dict, cfg_fields)
    if "attacks" in cfg_dict and "attack_specs" not in cfg_data:
        cfg_data["attack_specs"] = load_attacks_from_config(cfg_dict["attacks"])
    if "attack_specs" in cfg_data and isinstance(cfg_data["attack_specs"], list):
        cfg_data["attack_specs"] = load_attacks_from_config(cfg_data["attack_specs"])
    for key in ["external_eval_roots", "external_replay_roots"]:
        if key in cfg_data and cfg_data[key] is None:
            cfg_data[key] = ()
    cfg = ASRClosedLoopConfig(**cfg_data)
    out_dir = Path(str(cfg_dict.get("out") or "runs/asr_closed_loop_detox"))
    write_resolved_config(out_dir / "resolved_config.json", cfg_dict)
    manifest = run_asr_closed_loop_detox_yolo(
        model_path=cfg_dict["model"],
        images_dir=cfg_dict["images"],
        labels_dir=cfg_dict["labels"],
        data_yaml=cfg_dict["data_yaml"],
        target_classes=cfg_dict["target_classes"],
        output_dir=out_dir,
        cfg=cfg,
    )
    print(f"[DONE] status: {manifest.get('status')}")
    print(f"[DONE] final model: {manifest.get('final_model')}")
    print(f"[DONE] manifest: {out_dir / 'asr_closed_loop_detox_manifest.json'}")


if __name__ == "__main__":
    main()
