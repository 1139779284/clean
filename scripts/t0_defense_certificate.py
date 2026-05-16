#!/usr/bin/env python3
"""CLI for the T0 OD Defense Certificate.

Builds a publication-grade certificate for one or more defended models using:

* paired bootstrap confidence interval on per-image ASR reduction;
* Holm-Bonferroni step-down correction for family-wise error;
* Certified Minimum Reduction (CMR) ranking.

Manifest schema matches ``configs/t0_defense_leaderboard.example.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.defense_certificate import (
    DefenseCertificateConfig,
    build_defense_certificates,
    write_defense_certificates,
)
from model_security_gate.t0.defense_leaderboard import load_entries_from_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build T0 OD defense certificates with paired bootstrap and Holm-Bonferroni correction")
    p.add_argument("--manifest", required=True, help="JSON or YAML manifest listing defense entries")
    p.add_argument("--out", default="runs/t0_defense_certificate")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--fwer-alpha", type=float, default=0.05)
    p.add_argument("--max-clean-map-drop", type=float, default=0.03)
    p.add_argument("--min-certified-reduction", type=float, default=0.05)
    p.add_argument("--max-certified-asr", type=float, default=0.05)
    p.add_argument("--fail-on-no-certified", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = DefenseCertificateConfig(
        confidence=float(args.confidence),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        fwer_alpha=float(args.fwer_alpha),
        max_clean_map_drop=float(args.max_clean_map_drop),
        min_certified_reduction=float(args.min_certified_reduction),
        max_certified_asr=float(args.max_certified_asr),
    )
    entries = load_entries_from_manifest(args.manifest)
    if not entries:
        print("[ERROR] manifest has no entries", file=sys.stderr)
        return 2
    payload = build_defense_certificates(entries, cfg=cfg)
    json_path, md_path = write_defense_certificates(args.out, payload)
    print(f"[DONE] entries={payload['n_entries']} certified={payload['n_certified']}")
    print(f"[DONE] json: {json_path}")
    print(f"[DONE] report: {md_path}")
    if args.fail_on_no_certified and payload["n_certified"] == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
