#!/usr/bin/env python3
"""CLI: plan the contribution-1 Hybrid-PURIFY-OD ablation and emit a
ready-to-certify CFRC manifest.

Reads a YAML ablation spec, writes a runbook with exact training commands,
checks which artifacts already exist on disk, and builds the CFRC manifest
(same shape as ``scripts/t0_defense_certificate.py --manifest``) from any
completed arms.

Nothing is launched automatically; GPU runs are explicit so an autonomous
agent never burns hours without the user confirming.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.ablation_plan import (
    AblationArm,
    DetoxAblationSpec,
    build_cfrc_manifest,
    default_contribution_1_arms,
    plan_runs,
    write_ablation_plan,
)


def _opt_int(data: dict, key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(data: dict, key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_spec(path: str | Path) -> tuple[DetoxAblationSpec, list[AblationArm]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    spec_data = data.get("spec") or {}
    arms_data = data.get("arms")
    spec = DetoxAblationSpec(
        poisoned_model_id=str(spec_data["poisoned_model_id"]),
        poisoned_model_path=str(spec_data["poisoned_model_path"]),
        teacher_model=spec_data.get("teacher_model"),
        data_yaml=str(spec_data["data_yaml"]),
        images=str(spec_data["images"]),
        labels=str(spec_data["labels"]),
        target_classes=tuple(str(x) for x in spec_data.get("target_classes", [])),
        external_eval_roots=tuple(str(x) for x in spec_data.get("external_eval_roots", [])),
        external_replay_roots=tuple(str(x) for x in spec_data.get("external_replay_roots", [])),
        hybrid_config=(
            str(spec_data["hybrid_config"]) if spec_data.get("hybrid_config") else None
        ),
        imgsz=int(spec_data.get("imgsz", 640)),
        batch=int(spec_data.get("batch", 8)),
        cycles=int(spec_data.get("cycles", 3)),
        phase_epochs=int(spec_data.get("phase_epochs", 2)),
        feature_epochs=int(spec_data.get("feature_epochs", 2)),
        recovery_epochs=int(spec_data.get("recovery_epochs", 2)),
        device=str(spec_data.get("device", "0")),
        max_allowed_external_asr=float(spec_data.get("max_allowed_external_asr", 0.10)),
        max_map_drop=float(spec_data.get("max_map_drop", 0.03)),
        max_images=_opt_int(spec_data, "max_images"),
        eval_max_images=_opt_int(spec_data, "eval_max_images"),
        external_eval_max_images_per_attack=_opt_int(
            spec_data, "external_eval_max_images_per_attack"
        ),
        external_replay_max_images_per_attack=_opt_int(
            spec_data, "external_replay_max_images_per_attack"
        ),
        selection_max_map_drop=_opt_float(spec_data, "selection_max_map_drop"),
        no_pre_prune=bool(spec_data.get("no_pre_prune", False)),
        out_root=str(spec_data.get("out_root", "runs/t0_detox_ablation")),
        extra_cli=tuple(str(x) for x in spec_data.get("extra_cli", [])),
        notes=str(spec_data.get("notes", "")),
    )
    if arms_data:
        arms = [
            AblationArm(
                name=str(arm["name"]),
                use_lagrangian_controller=bool(arm.get("use_lagrangian_controller", False)),
                extra_cli=tuple(str(x) for x in arm.get("extra_cli", [])),
                config_overrides=dict(arm.get("config_overrides") or {}),
                notes=str(arm.get("notes", "")),
            )
            for arm in arms_data
        ]
    else:
        arms = list(default_contribution_1_arms())
    return spec, arms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plan Hybrid-PURIFY-OD contribution-1 ablation (static lambda vs "
            "Lagrangian lambda) and convert completed runs to a CFRC manifest"
        )
    )
    p.add_argument("--spec", required=True, help="YAML ablation spec")
    p.add_argument("--out", required=True, help="Output directory for plan and CFRC manifest")
    p.add_argument("--python", default="python", help="Python executable used in emitted train commands")
    p.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero if any arm is missing expected artifacts on disk",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    spec, arms = _load_spec(args.spec)
    runs = plan_runs(spec, arms, python=str(args.python))
    cfrc = build_cfrc_manifest(poisoned_model_id=spec.poisoned_model_id, runs=runs)
    out_paths = write_ablation_plan(
        args.out,
        spec=spec,
        arms=arms,
        runs=runs,
        cfrc_manifest=cfrc,
    )
    print(f"[DONE] plan: {out_paths['plan']}")
    print(f"[DONE] runbook: {out_paths['runbook']}")
    print(f"[DONE] cfrc manifest: {out_paths['cfrc_manifest']}")
    print(
        f"[DONE] arms={len(arms)} "
        f"completed={len(cfrc.get('entries') or [])} "
        f"pending={len(cfrc.get('skipped') or [])}"
    )
    if args.fail_on_missing and (cfrc.get("skipped") or []):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
