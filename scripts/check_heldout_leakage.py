from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from model_security_gate.utils.heldout_leakage import (
    find_path_leaks,
    scan_text_for_heldout_paths,
)


def load_heldout_roots(config_path: str | None, explicit_roots: list[str]) -> list[str]:
    roots = list(explicit_roots)
    if config_path:
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        roots.extend(str(item) for item in data.get("heldout_roots", []))
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if held-out test sets are used as detox/train inputs."
    )
    parser.add_argument(
        "--config",
        default="configs/heldout_sets.yaml",
        help="YAML file with heldout_roots list.",
    )
    parser.add_argument(
        "--heldout",
        action="append",
        default=[],
        help="Held-out root path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate train/detox input path to check. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="JSON/YAML/text manifest or resolved_config to scan for held-out paths.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config and Path(args.config).exists() else None
    heldout_roots = load_heldout_roots(config_path, args.heldout)
    if not heldout_roots:
        raise SystemExit("No heldout roots configured; pass --heldout or --config.")

    leaks = find_path_leaks(args.candidate, heldout_roots)
    for manifest in args.manifest:
        manifest_path = Path(manifest)
        if manifest_path.exists():
            text = manifest_path.read_text(encoding="utf-8", errors="ignore")
            leaks.extend(scan_text_for_heldout_paths(text, heldout_roots, str(manifest_path.resolve())))

    report = {
        "status": "failed" if leaks else "passed",
        "heldout_roots": [str(Path(root).expanduser().resolve()) for root in heldout_roots],
        "n_leaks": len(leaks),
        "leaks": [leak.__dict__ for leak in leaks],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())

