#!/usr/bin/env python3
"""CLI for the T0 OD defense leaderboard.

Ranks one or more (poisoned model, defended model) pairs using BackdoorBench-
style metrics adapted for object detection: per-attack Wilson 95% CI, paired
McNemar test, clean mAP50-95 drop, and a strict-dominance OD-DER score.

Manifest schema (YAML or JSON):

    entries:
      - name: hybrid_purify_v4_badnet_corner
        poisoned_model_id: badnet_oga_corner_pr2000_seed1
        defense: hybrid_purify_v4
        poisoned_external: runs/.../external_hard_suite_asr.json
        defended_external: runs/.../external_hard_suite_asr.json
        clean_before: runs/.../metrics.json
        clean_after:  runs/.../metrics.json
      - name: neural_cleanse_lite_badnet_corner
        poisoned_model_id: badnet_oga_corner_pr2000_seed1
        defense: neural_cleanse_lite
        poisoned_external: ...
        defended_external: ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.defense_leaderboard import (
    DefenseLeaderboardConfig,
    build_defense_leaderboard,
    load_entries_from_manifest,
    write_defense_leaderboard,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build T0 OD defense leaderboard from a manifest of defended runs")
    p.add_argument("--manifest", required=True, help="JSON or YAML manifest listing defense entries")
    p.add_argument("--out", default="runs/t0_defense_leaderboard")
    p.add_argument("--max-clean-map-drop", type=float, default=0.03)
    p.add_argument("--max-per-attack-regression", type=float, default=0.00)
    p.add_argument("--min-paired-sig-alpha", type=float, default=0.05)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--fail-on-no-accepted", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = DefenseLeaderboardConfig(
        max_clean_map_drop=float(args.max_clean_map_drop),
        max_per_attack_regression=float(args.max_per_attack_regression),
        min_paired_sig_alpha=float(args.min_paired_sig_alpha),
        confidence=float(args.confidence),
    )
    entries = load_entries_from_manifest(args.manifest)
    if not entries:
        print("[ERROR] manifest has no entries", file=sys.stderr)
        return 2
    leaderboard = build_defense_leaderboard(entries, cfg=cfg)
    json_path, md_path = write_defense_leaderboard(args.out, leaderboard)
    print(f"[DONE] entries={leaderboard['n_entries']} accepted={leaderboard['n_accepted']}")
    print(f"[DONE] json: {json_path}")
    print(f"[DONE] report: {md_path}")
    if args.fail_on_no_accepted and leaderboard["n_accepted"] == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
