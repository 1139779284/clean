#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.oda_score_calibration_repair import (
    ODAScoreCalibrationRepairConfig,
    run_oda_score_calibration_repair,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold-aware no-worse target-score calibration repair for residual ODA/semantic failures")
    p.add_argument("--config", default=None, help="YAML config. Accepts either top-level keys or oda_score_calibration_repair: {...}.")
    p.add_argument("--model", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--external-roots", nargs="+", default=None)
    p.add_argument("--target-classes", nargs="+", default=None)
    p.add_argument("--attack-names", nargs="*", default=None, help="Usually badnet_oda")
    p.add_argument("--failure-rows-csv", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--no-use-baseline-teacher", action="store_true", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--low-conf", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--letterbox-train", action="store_true", default=None, help="Use letterbox preprocessing in the repair dataloader.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-images-per-attack", type=int, default=None)
    p.add_argument("--replay-max-images-per-attack", type=int, default=None)
    p.add_argument("--failure-repeat", type=int, default=None)
    p.add_argument("--clean-anchor-images", type=int, default=None)
    p.add_argument("--guard-attack-names", nargs="*", default=None)
    p.add_argument("--guard-replay-max-images-per-attack", type=int, default=None)
    p.add_argument("--guard-repeat", type=int, default=None)
    p.add_argument("--guard-failure-only", action="store_true", default=None)
    p.add_argument("--lambda-score-calibration", type=float, default=None)
    p.add_argument("--lambda-task", type=float, default=None)
    p.add_argument("--lambda-oga-negative", type=float, default=None)
    p.add_argument("--lambda-semantic-negative", type=float, default=None)
    p.add_argument("--lambda-semantic-fp-region", type=float, default=None)
    p.add_argument("--lambda-oda-matched-anchor", type=float, default=None)
    p.add_argument("--lambda-oda-floor", type=float, default=None)
    p.add_argument("--lambda-target-absent-teacher-cap", type=float, default=None)
    p.add_argument("--score-conf-target", type=float, default=None)
    p.add_argument("--score-margin", type=float, default=None)
    p.add_argument("--score-topk-near", type=int, default=None)
    p.add_argument("--score-topk-far", type=int, default=None)
    p.add_argument("--score-positive-bce-weight", type=float, default=None)
    p.add_argument("--score-floor-weight", type=float, default=None)
    p.add_argument("--score-far-margin-weight", type=float, default=None)
    p.add_argument("--score-competing-margin-weight", type=float, default=None)
    p.add_argument("--score-teacher-weight", type=float, default=None)
    p.add_argument("--semantic-guard-keywords", nargs="*", default=None)
    p.add_argument("--semantic-negative-topk", type=int, default=None)
    p.add_argument("--semantic-negative-max-score", type=float, default=None)
    p.add_argument("--semantic-negative-margin-weight", type=float, default=None)
    p.add_argument("--semantic-negative-bce-weight", type=float, default=None)
    p.add_argument("--semantic-negative-active-margin", type=float, default=None)
    p.add_argument("--semantic-fp-region-topk", type=int, default=None)
    p.add_argument("--semantic-fp-region-iou-threshold", type=float, default=None)
    p.add_argument("--semantic-fp-region-center-radius", type=float, default=None)
    p.add_argument("--semantic-fp-region-max-score", type=float, default=None)
    p.add_argument("--semantic-fp-region-margin-weight", type=float, default=None)
    p.add_argument("--semantic-fp-region-bce-weight", type=float, default=None)
    p.add_argument("--semantic-fp-region-active-margin", type=float, default=None)
    p.add_argument("--target-absent-teacher-cap-topk", type=int, default=None)
    p.add_argument("--target-absent-teacher-cap-max-score", type=float, default=None)
    p.add_argument("--target-absent-teacher-cap-margin", type=float, default=None)
    p.add_argument("--oda-floor-min-score", type=float, default=None)
    p.add_argument("--oda-floor-teacher-margin", type=float, default=None)
    p.add_argument("--oda-matched-min-score", type=float, default=None)
    p.add_argument("--oda-matched-topk", type=int, default=None)
    p.add_argument(
        "--max-attack-asr",
        nargs="*",
        default=None,
        help="Hard per-attack ASR constraints, e.g. badnet_oda=0.05 semantic_green_cleanlabel=0.0.",
    )
    p.add_argument("--semantic-fp-required-max-conf", type=float, default=None)
    p.add_argument("--max-single-attack-worsen", type=float, default=None)
    p.add_argument("--max-allowed-external-asr", type=float, default=None)
    p.add_argument("--min-diagnostic-improvement", type=float, default=None)
    p.add_argument("--no-require-external-improvement", action="store_true", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    return p.parse_args()


def _parse_max_attack_asr(values: list[str] | None) -> dict[str, float] | None:
    if values is None:
        return None
    out: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"--max-attack-asr entries must be NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--max-attack-asr attack name is empty in {raw!r}")
        out[name] = float(value)
    return out


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    if "oda_score_calibration_repair" in data:
        data = data["oda_score_calibration_repair"] or {}
    if not isinstance(data, dict):
        raise ValueError("oda_score_calibration_repair config must be a mapping")
    return dict(data)


def _set_if(args: argparse.Namespace, cfg: dict[str, Any], arg_name: str, field_name: str | None = None) -> None:
    value = getattr(args, arg_name)
    if value is not None:
        cfg[field_name or arg_name] = value


def main() -> None:
    args = parse_args()
    values = _load_config(args.config)
    # CLI aliases and overrides.
    alias = {
        "data_yaml": "data_yaml",
        "out": "out_dir",
        "external_roots": "external_roots",
        "target_classes": "target_classes",
        "attack_names": "attack_names",
        "failure_rows_csv": "failure_rows_csv",
        "teacher_model": "teacher_model",
        "device": "device",
        "low_conf": "low_conf",
        "letterbox_train": "letterbox_train",
        "guard_attack_names": "guard_attack_names",
        "guard_replay_max_images_per_attack": "guard_replay_max_images_per_attack",
        "guard_failure_only": "guard_failure_only",
        "lambda_score_calibration": "lambda_score_calibration",
        "lambda_task": "lambda_task",
        "lambda_oga_negative": "lambda_oga_negative",
        "lambda_semantic_negative": "lambda_semantic_negative",
        "lambda_semantic_fp_region": "lambda_semantic_fp_region",
        "lambda_oda_matched_anchor": "lambda_oda_matched_anchor",
        "lambda_oda_floor": "lambda_oda_floor",
        "lambda_target_absent_teacher_cap": "lambda_target_absent_teacher_cap",
        "score_conf_target": "score_conf_target",
        "score_topk_near": "score_topk_near",
        "score_topk_far": "score_topk_far",
        "semantic_guard_keywords": "semantic_guard_keywords",
        "semantic_negative_topk": "semantic_negative_topk",
        "semantic_negative_max_score": "semantic_negative_max_score",
        "semantic_negative_margin_weight": "semantic_negative_margin_weight",
        "semantic_negative_bce_weight": "semantic_negative_bce_weight",
        "semantic_negative_active_margin": "semantic_negative_active_margin",
        "semantic_fp_region_topk": "semantic_fp_region_topk",
        "semantic_fp_region_iou_threshold": "semantic_fp_region_iou_threshold",
        "semantic_fp_region_center_radius": "semantic_fp_region_center_radius",
        "semantic_fp_region_max_score": "semantic_fp_region_max_score",
        "semantic_fp_region_margin_weight": "semantic_fp_region_margin_weight",
        "semantic_fp_region_bce_weight": "semantic_fp_region_bce_weight",
        "semantic_fp_region_active_margin": "semantic_fp_region_active_margin",
        "target_absent_teacher_cap_topk": "target_absent_teacher_cap_topk",
        "target_absent_teacher_cap_max_score": "target_absent_teacher_cap_max_score",
        "target_absent_teacher_cap_margin": "target_absent_teacher_cap_margin",
        "oda_floor_min_score": "oda_floor_min_score",
        "oda_floor_teacher_margin": "oda_floor_teacher_margin",
        "oda_matched_min_score": "oda_matched_min_score",
        "oda_matched_topk": "oda_matched_topk",
        "semantic_fp_required_max_conf": "semantic_fp_required_max_conf",
        "max_single_attack_worsen": "max_single_attack_worsen",
        "max_allowed_external_asr": "max_allowed_external_asr",
        "min_diagnostic_improvement": "min_diag_score_improvement",
    }
    for name in [
        "model", "imgsz", "conf", "batch", "epochs", "lr", "weight_decay", "max_images_per_attack",
        "replay_max_images_per_attack", "failure_repeat", "clean_anchor_images", "guard_repeat",
        "score_margin", "score_positive_bce_weight", "score_floor_weight", "score_far_margin_weight",
        "score_competing_margin_weight", "score_teacher_weight", "amp",
    ]:
        _set_if(args, values, name)
    for arg_name, field_name in alias.items():
        _set_if(args, values, arg_name, field_name)
    if args.max_attack_asr is not None:
        values["max_attack_asr"] = _parse_max_attack_asr(args.max_attack_asr)
    if args.no_use_baseline_teacher:
        values["use_baseline_teacher"] = False
    if args.no_require_external_improvement:
        values["require_external_improvement_for_final"] = False

    valid_fields = {f.name for f in fields(ODAScoreCalibrationRepairConfig)}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown ODAScoreCalibrationRepairConfig keys: {unknown}")
    missing = [name for name in ("model", "data_yaml", "out_dir") if not values.get(name)]
    if not values.get("external_roots"):
        missing.append("external_roots")
    if not values.get("target_classes"):
        missing.append("target_classes")
    if missing:
        raise ValueError(f"Missing required config/CLI values: {missing}")
    cfg = ODAScoreCalibrationRepairConfig(**values)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = run_oda_score_calibration_repair(cfg)
    print(json.dumps({k: manifest.get(k) for k in ("status", "rolled_back", "final_model", "best", "best_by_diagnostic")}, indent=2, ensure_ascii=False))
    print(f"[DONE] manifest: {out / 'oda_score_calibration_repair_manifest.json'}")


if __name__ == "__main__":
    main()
