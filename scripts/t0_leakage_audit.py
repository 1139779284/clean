#!/usr/bin/env python3
"""CLI for T0 train-eval leakage audit.

Reads a CFRC manifest plus (optionally) a mapping of arm -> Hybrid-PURIFY
manifest and writes a severity-ranked audit JSON/Markdown.  Use before
citing CFRC results in a paper to confirm external replay and external
evaluation are disjoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.t0.leakage_audit import audit_cfrc_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit CFRC entries for external replay/eval leakage")
    p.add_argument("--cfrc-manifest", required=True)
    p.add_argument(
        "--hybrid-manifest",
        action="append",
        default=[],
        help="<arm_name>=<path> entries; may be repeated",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--fail-on-blocked", action="store_true")
    p.add_argument("--fail-on-warn", action="store_true")
    return p.parse_args()


def _parse_mapping(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            continue
        arm, path = item.split("=", 1)
        arm = arm.strip()
        if arm:
            out[arm] = path.strip()
    return out


def _render_markdown(report: dict) -> str:
    worst = report.get("worst_severity", "ok")
    lines = ["# T0 Leakage Audit", "", f"- worst severity: `{worst}`", ""]
    lines.append("| arm | severity | same roots | shared attacks | recommendation |")
    lines.append("|---|---|---|---|---|")
    for row in report.get("entries", []):
        same = ", ".join(row.get("train_eval_same_roots") or []) or "-"
        shared = ", ".join(row.get("shared_attack_keys") or []) or "-"
        lines.append(
            f"| `{row.get('arm')}` | `{row.get('severity')}` | {same} | {shared} | "
            f"{row.get('recommendation')} |"
        )
    lines.append("")
    for row in report.get("entries", []):
        notes = row.get("notes") or []
        if not notes:
            continue
        lines.append(f"Notes for `{row.get('arm')}`:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    cfrc_path = Path(args.cfrc_manifest)
    text = cfrc_path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        manifest = yaml.safe_load(text) or {}
    mapping = _parse_mapping(list(args.hybrid_manifest))
    report = audit_cfrc_manifest(cfrc_manifest=manifest, manifests_by_arm=mapping)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "t0_leakage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "T0_LEAKAGE_AUDIT.md").write_text(_render_markdown(report), encoding="utf-8")
    worst = report.get("worst_severity", "ok")
    print(f"[DONE] worst severity: {worst}")
    print(f"[DONE] json: {out / 't0_leakage_audit.json'}")
    print(f"[DONE] report: {out / 'T0_LEAKAGE_AUDIT.md'}")
    if args.fail_on_blocked and worst == "blocked":
        return 3
    if args.fail_on_warn and worst in {"warn", "blocked"}:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
