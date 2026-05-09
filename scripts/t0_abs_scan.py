#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.scan.abs import detect_abs_suspicious_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight ABS-style channel scoring from exported activations")
    parser.add_argument("--input", required=True, help="JSON with `activations` and `target_scores` arrays")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = detect_abs_suspicious_channels(
        data.get("activations") or [],
        data.get("target_scores") or [],
        top_fraction=float(args.top_fraction),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] suspicious_channels={len(result.suspicious_channels)} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
