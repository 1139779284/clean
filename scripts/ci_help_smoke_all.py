#!/usr/bin/env python3
"""Run --help smoke checks for all scripts.

Use --allow-missing-heavy-deps only in intentionally lightweight CI jobs that do
not install torch/ultralytics. In full pixi/production CI, run without that flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List

HEAVY_DEP_MARKERS = (
    "ModuleNotFoundError: No module named 'torch'",
    "ModuleNotFoundError: No module named 'ultralytics'",
    "ImportError: No module named torch",
    "ImportError: No module named ultralytics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run python scripts/*.py --help for every script")
    parser.add_argument("--scripts-dir", default="scripts")
    parser.add_argument("--pattern", default="*.py")
    parser.add_argument("--exclude", action="append", default=[], help="Script basename to exclude; repeatable")
    parser.add_argument("--allow-missing-heavy-deps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scripts_dir = Path(args.scripts_dir)
    excluded = set(args.exclude)
    failures: List[str] = []
    warnings: List[str] = []

    scripts = sorted(p for p in scripts_dir.glob(args.pattern) if p.is_file() and p.name not in excluded and not p.name.startswith("_"))
    if not scripts:
        print(f"No scripts found in {scripts_dir}", file=sys.stderr)
        return 1

    for script in scripts:
        proc = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True)
        if proc.returncode == 0:
            print(f"[OK] {script}")
            continue
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
        if args.allow_missing_heavy_deps and any(marker in combined for marker in HEAVY_DEP_MARKERS):
            warnings.append(f"[WARN] {script}: missing heavy dependency in lightweight CI")
            print(warnings[-1])
            continue
        failures.append(f"[FAIL] {script}: exit={proc.returncode}\n{combined[-1200:]}")

    if warnings:
        print("\n".join(warnings), file=sys.stderr)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
