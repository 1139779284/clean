#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.pareto_merge import (
    generate_group_layer_alpha_specs,
    interpolate_checkpoints,
    parse_alpha_grid,
    parse_layer_alpha_spec,
    parse_named_layer_alpha_specs,
    write_merge_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search Pareto YOLO weight merges between a mAP-preserving and ASR-suppressing checkpoint")
    p.add_argument("--base-model", required=True, help="mAP-preserving / balanced checkpoint. alpha=0 keeps this model.")
    p.add_argument("--source-model", required=True, help="ASR-suppressing / strong checkpoint. alpha=1 keeps this model.")
    p.add_argument("--out", required=True)
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    p.add_argument("--layer-alpha-spec", default=None, help="Optional YOLO layer alpha ranges, e.g. '0-9:0.2,10-21:0.5,22-999:0.8'.")
    p.add_argument("--layer-alpha-specs", default=None, help="Pipe-separated fixed layer specs, e.g. 'head_high::0-9:0.1,10-21:0.3,22-999:0.8|backbone_high::0-9:0.8,10-999:0.2'.")
    p.add_argument("--layer-default-alpha", type=float, default=0.0, help="Default alpha for fixed layer specs when a tensor key is outside all specified ranges.")
    p.add_argument("--layer-grid-alphas", default=None, help="Generate coarse backbone/neck/head layer specs from these alphas, e.g. '0,0.25,0.5,0.75,1'.")
    p.add_argument("--layer-grid-backbone", default="0-9")
    p.add_argument("--layer-grid-neck", default="10-21")
    p.add_argument("--layer-grid-head", default="22-999")
    p.add_argument("--max-layer-candidates", type=int, default=40)
    p.add_argument("--prefix", default="pareto")
    p.add_argument("--eval-data-yaml", default=None)
    p.add_argument("--eval-external-roots", nargs="*", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images-per-attack", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--score-max-weight", type=float, default=1.0)
    p.add_argument("--score-mean-weight", type=float, default=0.35)
    p.add_argument("--score-map-weight", type=float, default=0.20)
    p.add_argument("--max-allowed-external-asr", type=float, default=None)
    p.add_argument("--max-map-drop", type=float, default=None)
    p.add_argument("--baseline-map50-95", type=float, default=None)
    p.add_argument("--min-map50-95", type=float, default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-clean-eval", action="store_true")
    p.add_argument("--skip-external-eval", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_score(clean: dict[str, Any], external: dict[str, Any], args: argparse.Namespace) -> float:
    ext_summary = external.get("summary") or {}
    max_asr = float(ext_summary.get("max_asr", 1.0))
    mean_asr = float(ext_summary.get("mean_asr", max_asr))
    map50_95 = float(clean.get("map50_95", 0.0))
    return float(args.score_max_weight) * max_asr + float(args.score_mean_weight) * mean_asr - float(args.score_map_weight) * map50_95


def _accepted(row: dict[str, Any], args: argparse.Namespace, baseline_map50_95: float | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    max_asr = _as_float(row.get("external_max_asr"))
    map50_95 = _as_float(row.get("map50_95"))
    if args.max_allowed_external_asr is not None:
        if max_asr is None:
            reasons.append("missing_external_asr")
        elif max_asr > float(args.max_allowed_external_asr):
            reasons.append("external_asr_too_high")
    if args.min_map50_95 is not None:
        if map50_95 is None:
            reasons.append("missing_map50_95")
        elif map50_95 < float(args.min_map50_95):
            reasons.append("map50_95_too_low")
    if args.max_map_drop is not None and baseline_map50_95 is not None:
        if map50_95 is None:
            reasons.append("missing_map50_95")
        else:
            drop = float(baseline_map50_95) - float(map50_95)
            row["map50_95_drop"] = drop
            if drop > float(args.max_map_drop):
                reasons.append("map_drop_too_high")
    return len(reasons) == 0, reasons


def _candidate_id(prefix: str, name: str, alpha: float) -> str:
    safe_alpha = str(float(alpha)).replace(".", "p")
    safe_name = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in name).strip("_") or "candidate"
    return f"{prefix}_{safe_name}_alpha_{safe_alpha}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_alpha_grid(args.alphas)
    layer_spec = parse_layer_alpha_spec(args.layer_alpha_spec)

    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        name = "global" if not layer_spec else "layer_spec"
        candidates.append({"name": name, "alpha": float(alpha), "alpha_by_layer": layer_spec})

    for named in parse_named_layer_alpha_specs(args.layer_alpha_specs):
        candidates.append({"name": named.name, "alpha": float(args.layer_default_alpha), "alpha_by_layer": named.alpha_by_layer})

    if args.layer_grid_alphas:
        grid_specs = generate_group_layer_alpha_specs(
            parse_alpha_grid(args.layer_grid_alphas),
            backbone_range=args.layer_grid_backbone,
            neck_range=args.layer_grid_neck,
            head_range=args.layer_grid_head,
            max_candidates=args.max_layer_candidates,
        )
        for named in grid_specs:
            candidates.append({"name": f"grid_{named.name}", "alpha": float(args.layer_default_alpha), "alpha_by_layer": named.alpha_by_layer})

    reports = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, tuple[tuple[str, float], ...]]] = set()
    for candidate in candidates:
        alpha = float(candidate["alpha"])
        alpha_by_layer = dict(candidate.get("alpha_by_layer") or {})
        signature = (alpha, tuple(sorted((str(k), float(v)) for k, v in alpha_by_layer.items())))
        if signature in seen:
            continue
        seen.add(signature)
        candidate_id = _candidate_id(args.prefix, str(candidate["name"]), alpha)
        model_path = out_dir / "models" / f"{candidate_id}.pt"
        report = interpolate_checkpoints(
            base_model=args.base_model,
            source_model=args.source_model,
            output_model=model_path,
            alpha=alpha,
            alpha_by_layer=alpha_by_layer,
        )
        reports.append(report)
        clean_json: dict[str, Any] = {}
        external_json: dict[str, Any] = {}
        if not args.skip_eval:
            if args.eval_data_yaml and not args.skip_clean_eval:
                clean_out = out_dir / "eval" / candidate_id / "clean_metrics.json"
                clean_out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "eval_yolo_metrics.py"),
                    "--model",
                    str(model_path),
                    "--data-yaml",
                    str(args.eval_data_yaml),
                    "--out",
                    str(clean_out),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--workers",
                    "0",
                ]
                if args.device is not None:
                    cmd += ["--device", str(args.device)]
                _run(cmd)
                clean_json = _read_json(clean_out)
            if not args.skip_external_eval and args.eval_external_roots and args.eval_data_yaml and args.target_classes:
                ext_out = out_dir / "eval" / candidate_id / "external"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "run_external_hard_suite.py"),
                    "--model",
                    str(model_path),
                    "--data-yaml",
                    str(args.eval_data_yaml),
                    "--out",
                    str(ext_out),
                    "--imgsz",
                    str(args.imgsz),
                    "--conf",
                    str(args.conf),
                    "--max-images-per-attack",
                    str(args.max_images_per_attack),
                    "--target-classes",
                    *[str(x) for x in args.target_classes],
                    "--roots",
                    *[str(x) for x in args.eval_external_roots],
                ]
                if args.device is not None:
                    cmd += ["--device", str(args.device)]
                _run(cmd)
                external_json = _read_json(ext_out / "external_hard_suite_asr.json")

        ext_summary = external_json.get("summary") or {}
        row = {
            "alpha": float(alpha),
            "candidate_id": candidate_id,
            "candidate_name": str(candidate["name"]),
            "alpha_by_layer": json.dumps(alpha_by_layer, sort_keys=True),
            "model": str(model_path),
            "map50": clean_json.get("map50"),
            "map50_95": clean_json.get("map50_95"),
            "precision": clean_json.get("precision"),
            "recall": clean_json.get("recall"),
            "external_max_asr": ext_summary.get("max_asr"),
            "external_mean_asr": ext_summary.get("mean_asr"),
            "score": _candidate_score(clean_json, external_json, args) if external_json else None,
        }
        matrix = ext_summary.get("asr_matrix") or {}
        for key, value in matrix.items():
            row[f"asr::{key}"] = value
        rows.append(row)
        print("[CANDIDATE]", json.dumps(row, ensure_ascii=False), flush=True)

    manifest = write_merge_manifest(
        out_dir / "pareto_merge_manifest.json",
        reports,
        extra={
            "base_model": args.base_model,
            "source_model": args.source_model,
            "alphas": alphas,
            "layer_alpha_spec": layer_spec,
            "layer_alpha_specs": args.layer_alpha_specs,
            "layer_grid_alphas": args.layer_grid_alphas,
            "rows_csv": str(out_dir / "pareto_merge_results.csv"),
        },
    )

    baseline_map50_95 = args.baseline_map50_95
    if baseline_map50_95 is None:
        for row in rows:
            if float(row.get("alpha") or 0.0) == 0.0:
                baseline_map50_95 = _as_float(row.get("map50_95"))
                if baseline_map50_95 is not None:
                    break
    for row in rows:
        ok, reasons = _accepted(row, args, baseline_map50_95)
        row["accepted"] = ok
        row["reject_reasons"] = ";".join(reasons)
        if "map50_95_drop" not in row and baseline_map50_95 is not None and _as_float(row.get("map50_95")) is not None:
            row["map50_95_drop"] = float(baseline_map50_95) - float(row["map50_95"])

    scored_rows = [r for r in rows if _as_float(r.get("score")) is not None]
    best_by_score = min(scored_rows, key=lambda r: float(r["score"])) if scored_rows else None
    accepted_rows = [r for r in scored_rows if r.get("accepted")]
    best_accepted = min(accepted_rows, key=lambda r: float(r["score"])) if accepted_rows else None
    best_payload = {
        "best_model": (best_accepted or best_by_score or {}).get("model"),
        "best_accepted": best_accepted,
        "best_by_score": best_by_score,
        "n_candidates": len(rows),
        "n_accepted": len(accepted_rows),
        "baseline_map50_95": baseline_map50_95,
        "criteria": {
            "max_allowed_external_asr": args.max_allowed_external_asr,
            "max_map_drop": args.max_map_drop,
            "min_map50_95": args.min_map50_95,
            "score_max_weight": args.score_max_weight,
            "score_mean_weight": args.score_mean_weight,
            "score_map_weight": args.score_map_weight,
        },
    }
    (out_dir / "pareto_merge_best.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "pareto_merge_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] manifest: {manifest}")
    print(f"[DONE] results: {out_dir / 'pareto_merge_results.csv'}")
    print(f"[DONE] best: {out_dir / 'pareto_merge_best.json'}")


if __name__ == "__main__":
    main()
