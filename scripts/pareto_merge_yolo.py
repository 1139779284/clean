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
    interpolate_checkpoints,
    parse_alpha_grid,
    parse_layer_alpha_spec,
    write_merge_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search Pareto YOLO weight merges between a mAP-preserving and ASR-suppressing checkpoint")
    p.add_argument("--base-model", required=True, help="mAP-preserving / balanced checkpoint. alpha=0 keeps this model.")
    p.add_argument("--source-model", required=True, help="ASR-suppressing / strong checkpoint. alpha=1 keeps this model.")
    p.add_argument("--out", required=True)
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    p.add_argument("--layer-alpha-spec", default=None, help="Optional YOLO layer alpha ranges, e.g. '0-9:0.2,10-21:0.5,22-999:0.8'.")
    p.add_argument("--prefix", default="pareto")
    p.add_argument("--eval-data-yaml", default=None)
    p.add_argument("--eval-external-roots", nargs="*", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images-per-attack", type=int, default=0)
    p.add_argument("--device", default=None)
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


def _candidate_score(clean: dict[str, Any], external: dict[str, Any]) -> float:
    ext_summary = external.get("summary") or {}
    max_asr = float(ext_summary.get("max_asr", 1.0))
    mean_asr = float(ext_summary.get("mean_asr", max_asr))
    map50_95 = float(clean.get("map50_95", 0.0))
    return max_asr + 0.35 * mean_asr - 0.20 * map50_95


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_alpha_grid(args.alphas)
    layer_spec = parse_layer_alpha_spec(args.layer_alpha_spec)

    reports = []
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        safe_alpha = str(float(alpha)).replace(".", "p")
        model_path = out_dir / "models" / f"{args.prefix}_alpha_{safe_alpha}.pt"
        report = interpolate_checkpoints(
            base_model=args.base_model,
            source_model=args.source_model,
            output_model=model_path,
            alpha=float(alpha),
            alpha_by_layer=layer_spec,
        )
        reports.append(report)
        clean_json: dict[str, Any] = {}
        external_json: dict[str, Any] = {}
        if not args.skip_eval:
            if args.eval_data_yaml and not args.skip_clean_eval:
                clean_out = out_dir / "eval" / f"alpha_{safe_alpha}" / "clean_metrics.json"
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
                ext_out = out_dir / "eval" / f"alpha_{safe_alpha}" / "external"
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
            "model": str(model_path),
            "map50": clean_json.get("map50"),
            "map50_95": clean_json.get("map50_95"),
            "precision": clean_json.get("precision"),
            "recall": clean_json.get("recall"),
            "external_max_asr": ext_summary.get("max_asr"),
            "external_mean_asr": ext_summary.get("mean_asr"),
            "score": _candidate_score(clean_json, external_json) if external_json else None,
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
            "rows_csv": str(out_dir / "pareto_merge_results.csv"),
        },
    )
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "pareto_merge_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] manifest: {manifest}")
    print(f"[DONE] results: {out_dir / 'pareto_merge_results.csv'}")


if __name__ == "__main__":
    main()
