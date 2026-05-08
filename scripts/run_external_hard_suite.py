#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.external_hard_suite import ExternalHardSuiteConfig, run_external_hard_suite_for_yolo, write_external_hard_suite_outputs
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a YOLO model on external hard-suite attack datasets")
    p.add_argument("--config", default=None, help="YAML config. Values under `external_hard_suite:` are accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--roots", nargs="*", default=None, help="Benchmark roots to discover, e.g. poison_benchmark_cuda_large")
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--match-iou", type=float, default=None)
    p.add_argument(
        "--oda-success-mode",
        choices=["localized_any_recalled", "class_presence", "strict_all_recalled"],
        default=None,
        help=(
            "ODA ASR definition: localized_any_recalled=current default, "
            "class_presence=success only if no target-class detection exists, "
            "strict_all_recalled=success if any GT target is missing."
        ),
    )
    p.add_argument("--max-images-per-attack", type=int, default=None)
    p.add_argument("--semantic-abstain-rules", default=None, help="YAML/JSON semantic runtime-abstain rules to apply before scoring target-absent semantic rows.")
    p.add_argument("--apply-semantic-abstain", action="store_true", default=None, help="Apply semantic runtime-abstain rules during guarded external evaluation.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = {
        "model": None,
        "data_yaml": None,
        "target_classes": None,
        "roots": (),
        "out": "runs/external_hard_suite",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "match_iou": 0.30,
        "oda_success_mode": "localized_any_recalled",
        "max_images_per_attack": 0,
        "semantic_abstain_rules": None,
        "apply_semantic_abstain": False,
        "device": None,
    }
    raw = load_yaml_config(args.config, section="external_hard_suite")
    resolved = deep_merge(defaults, deep_merge(raw, namespace_overrides(args, exclude={"config"})))
    missing = [k for k in ["model", "data_yaml", "target_classes", "roots"] if not resolved.get(k)]
    if missing:
        raise SystemExit(f"Missing required values: {', '.join(missing)}")
    out_dir = Path(str(resolved.get("out") or "runs/external_hard_suite"))
    write_resolved_config(out_dir / "resolved_config.json", resolved)
    cfg = ExternalHardSuiteConfig(
        roots=tuple(resolved.get("roots") or ()),
        conf=float(resolved.get("conf", 0.25)),
        iou=float(resolved.get("iou", 0.7)),
        imgsz=int(resolved.get("imgsz", 640)),
        match_iou=float(resolved.get("match_iou", 0.30)),
        oda_success_mode=str(resolved.get("oda_success_mode", "localized_any_recalled")),
        max_images_per_attack=int(resolved.get("max_images_per_attack", 0) or 0),
        semantic_abstain_rules=resolved.get("semantic_abstain_rules"),
        apply_semantic_abstain=bool(resolved.get("apply_semantic_abstain", False)),
    )
    result = run_external_hard_suite_for_yolo(
        model_path=resolved["model"],
        data_yaml=resolved["data_yaml"],
        target_classes=resolved["target_classes"],
        cfg=cfg,
        device=resolved.get("device"),
    )
    json_path, rows_path = write_external_hard_suite_outputs(result, out_dir)
    print(f"[DONE] max_asr={result.get('summary', {}).get('max_asr')} mean_asr={result.get('summary', {}).get('mean_asr')}")
    print(f"[DONE] report: {json_path}")
    print(f"[DONE] rows: {rows_path}")


if __name__ == "__main__":
    main()
