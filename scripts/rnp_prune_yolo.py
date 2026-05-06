#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.rnp import RNPConfig, apply_rnp_soft_suppression, score_rnp_channels_for_yolo
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, split_known_keys, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RNP-lite channel scoring and conservative soft suppression for YOLO")
    p.add_argument("--config", default=None, help="YAML config; section rnp is accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--out", default=None, help="Output .pt path for soft-suppressed model")
    p.add_argument("--score-csv", default=None, help="Optional existing score CSV; if absent, scoring is run first")
    p.add_argument("--score-out", default=None, help="Where to write RNP score CSV")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--unlearn-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--score-only", action="store_true", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = {"model": None, "data_yaml": None, "out": "runs/rnp_soft_pruned.pt", "score_out": "runs/rnp_scores.csv"}
    defaults.update(RNPConfig().__dict__)
    raw = load_yaml_config(args.config, section="rnp")
    resolved = deep_merge(defaults, deep_merge(raw, namespace_overrides(args, exclude={"config"})))
    missing = [k for k in ["model", "data_yaml"] if not resolved.get(k)]
    if missing:
        raise SystemExit("Missing required values: " + ", ".join(missing))
    out_path = Path(str(resolved.get("out") or "runs/rnp_soft_pruned.pt"))
    write_resolved_config(out_path.with_suffix(".resolved_config.json"), resolved)
    cfg_keys = set(RNPConfig.__dataclass_fields__.keys())
    cfg_data, _extra = split_known_keys(resolved, cfg_keys)
    cfg = RNPConfig(**cfg_data)
    score_csv = resolved.get("score_csv")
    if not score_csv:
        score_csv, summary = score_rnp_channels_for_yolo(resolved["model"], resolved["data_yaml"], resolved.get("score_out") or out_path.with_suffix(".scores.csv"), cfg)
        print(f"[DONE] scores: {score_csv}")
        print(f"[DONE] summary: {summary}")
    if resolved.get("score_only"):
        return
    output = apply_rnp_soft_suppression(
        model_path=resolved["model"],
        score_csv=score_csv,
        output_path=out_path,
        top_k=int(resolved.get("top_k") or cfg.score_top_k),
        strength=float(resolved.get("strength") or cfg.soft_suppression_strength),
        min_score=float(resolved.get("min_score") or cfg.min_score_to_prune),
        device=resolved.get("device"),
    )
    print(f"[DONE] soft-pruned model: {output}")


if __name__ == "__main__":
    main()
