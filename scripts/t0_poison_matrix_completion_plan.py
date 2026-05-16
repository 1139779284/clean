#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.poison_matrix_evidence import PoisonMatrixEvidenceConfig, build_poison_matrix_evidence


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_attack_epochs(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in _split_csv(raw):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = int(value)
    return out


def _load_config(path: str | None, args: argparse.Namespace) -> PoisonMatrixEvidenceConfig:
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    expected_attacks = data.get("expected_attacks")
    expected_seeds = data.get("expected_seeds")
    expected_poison_rates = data.get("expected_poison_rates")
    return PoisonMatrixEvidenceConfig(
        expected_attacks=tuple(str(x) for x in expected_attacks) if isinstance(expected_attacks, list) else tuple(_split_csv(args.expected_attacks)),
        expected_seeds=tuple(int(x) for x in expected_seeds) if isinstance(expected_seeds, list) else tuple(int(x) for x in _split_csv(args.expected_seeds)),
        expected_poison_rates=tuple(float(x) for x in expected_poison_rates) if isinstance(expected_poison_rates, list) else tuple(float(x) for x in _split_csv(args.expected_poison_rates)),
        min_primary_asr=float(data.get("min_primary_asr", args.min_primary_asr)),
        min_usable_asr=float(data.get("min_usable_asr", args.min_usable_asr)),
        require_weights=bool(data.get("require_weights", True)),
        require_report=bool(data.get("require_report", True)),
        require_any_strong=bool(data.get("require_any_strong", False)),
        require_full_factorial=bool(data.get("require_full_factorial", True)),
        full_factorial_cell_acceptance=str(data.get("full_factorial_cell_acceptance", "present")),
    )


def _run_name(attack: str, poison_rate: float, seed: int) -> str:
    return f"{attack}_pr{int(round(float(poison_rate) * 10000)):04d}_seed{int(seed)}"


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def _command(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if any(ch.isspace() for ch in part) else part for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute missing T0 poison-model matrix cells")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--evidence-config", default="configs/t0_poison_matrix_full_evidence.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="runs/t0_poison_matrix_completion_plan")
    parser.add_argument("--train-out", default="runs/t0_poison_matrix_completion")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--attack-config", default="configs/t0_poison_core_attacks.yaml")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--eval-roots", required=True)
    parser.add_argument("--target-classes", default="helmet")
    parser.add_argument("--expected-attacks", default="badnet_oga_corner,semantic_cleanlabel,wanet_oga")
    parser.add_argument("--expected-seeds", default="1,2,3")
    parser.add_argument("--expected-poison-rates", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--min-primary-asr", type=float, default=0.20)
    parser.add_argument("--min-usable-asr", type=float, default=0.05)
    parser.add_argument("--default-epochs", type=int, default=5)
    parser.add_argument("--attack-epochs", default="wanet_oga=10")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.evidence_config, args)
    evidence = build_poison_matrix_evidence(summary_json=args.summary_json, root=args.root, cfg=cfg)
    attack_epochs = _parse_attack_epochs(args.attack_epochs)
    out_dir = Path(args.out)
    train_out = Path(args.train_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, Any]] = []
    for attack, row in (evidence.get("coverage") or {}).items():
        for cell in row.get("missing_cells") or []:
            missing.append({"attack": attack, "seed": int(cell["seed"]), "poison_rate": float(cell["poison_rate"])})
    if args.max_cells > 0:
        missing = missing[: int(args.max_cells)]

    cells: list[dict[str, Any]] = []
    merged_entries = []
    if args.summary_json and Path(args.summary_json).exists():
        existing = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        merged_entries.extend(existing.get("entries") or [])

    for cell in missing:
        attack = str(cell["attack"])
        seed = int(cell["seed"])
        poison_rate = float(cell["poison_rate"])
        run = _run_name(attack, poison_rate, seed)
        epochs = int(attack_epochs.get(attack, args.default_epochs))
        weights = train_out / "training" / run / "weights" / "best.pt"
        report = train_out / "eval" / run / "external_hard_suite_asr.json"
        train_cmd = [
            sys.executable,
            "scripts/train_t0_poison_models_yolo.py",
            "--clean-root",
            args.clean_root,
            "--base-model",
            args.base_model,
            "--out",
            _rel(train_out),
            "--attack-config",
            args.attack_config,
            "--attacks",
            attack,
            "--poison-rates",
            str(poison_rate),
            "--seeds",
            str(seed),
            "--epochs",
            str(epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--device",
            str(args.device),
        ]
        if args.skip_existing:
            train_cmd.append("--skip-existing")
        eval_cmd = [
            sys.executable,
            "scripts/run_external_hard_suite.py",
            "--model",
            _rel(weights),
            "--data-yaml",
            args.data_yaml,
            "--target-classes",
            *args.target_classes.split(),
            "--roots",
            args.eval_roots,
            "--out",
            _rel(report.parent),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            "0.25",
            "--iou",
            "0.7",
            "--match-iou",
            "0.3",
            "--oda-success-mode",
            "localized_any_recalled",
            "--device",
            str(args.device),
        ]
        entry = {
            "attack": attack,
            "run": run,
            "poison_rate": poison_rate,
            "seed": seed,
            "epochs": epochs,
            "weights": _rel(weights),
            "report": _rel(report),
        }
        merged_entries.append(entry)
        cells.append(
            {
                **entry,
                "train_command": _command(train_cmd),
                "eval_command": _command(eval_cmd),
                "train_argv": train_cmd,
                "eval_argv": eval_cmd,
                "weights_exists": weights.exists(),
                "report_exists": report.exists(),
            }
        )

    plan = {
        "status": "ready" if cells else "no_missing_cells",
        "source_evidence_status": evidence.get("status"),
        "n_missing_total": sum(len(row.get("missing_cells") or []) for row in (evidence.get("coverage") or {}).values()),
        "n_planned": len(cells),
        "execute": bool(args.execute),
        "cells": cells,
    }
    (out_dir / "t0_poison_matrix_completion_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "t0_poison_matrix_merged_summary_planned.json").write_text(
        json.dumps({"entries": merged_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["# T0 Poison Matrix Completion Plan", ""]
    lines.append(f"- source evidence status: `{evidence.get('status')}`")
    lines.append(f"- missing cells total: `{plan['n_missing_total']}`")
    lines.append(f"- planned cells: `{len(cells)}`")
    lines.append("")
    lines.append("| attack | poison rate | seed | epochs | weights |")
    lines.append("|---|---:|---:|---:|---|")
    for cell in cells:
        lines.append(f"| `{cell['attack']}` | `{cell['poison_rate']}` | `{cell['seed']}` | `{cell['epochs']}` | `{cell['weights']}` |")
    (out_dir / "T0_POISON_MATRIX_COMPLETION_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.execute:
        for cell in cells:
            print(f"[TRAIN] {cell['run']}")
            subprocess.run(cell["train_argv"], check=True)
            print(f"[EVAL] {cell['run']}")
            subprocess.run(cell["eval_argv"], check=True)

    print(f"[DONE] planned={len(cells)} total_missing={plan['n_missing_total']}")
    print(f"[DONE] plan: {out_dir / 'T0_POISON_MATRIX_COMPLETION_PLAN.md'}")
    print(f"[DONE] merged summary: {out_dir / 't0_poison_matrix_merged_summary_planned.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
