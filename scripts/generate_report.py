#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.report.report_generator import generate_html_report, generate_markdown_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Markdown/HTML report from Model Security Gate artifacts")
    p.add_argument("--security-report", default=None)
    p.add_argument("--before-report", default=None)
    p.add_argument("--after-report", default=None)
    p.add_argument("--before-metrics", default=None)
    p.add_argument("--after-metrics", default=None)
    p.add_argument("--pseudo-quality", default=None, help="pseudo_label_manifest.json or quality summary JSON")
    p.add_argument("--acceptance", default=None)
    p.add_argument("--detox-manifest", default=None, help="Optional strong_detox_manifest.json for supervision/verification context")
    p.add_argument("--scan-dir", default=None)
    p.add_argument("--out-md", default="runs/model_security_report.md")
    p.add_argument("--out-html", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    md = generate_markdown_report(
        security_report=args.security_report,
        before_report=args.before_report,
        after_report=args.after_report,
        before_metrics=args.before_metrics,
        after_metrics=args.after_metrics,
        pseudo_quality=args.pseudo_quality,
        acceptance=args.acceptance,
        detox_manifest=args.detox_manifest,
        scan_dir=args.scan_dir,
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"[DONE] wrote {out_md}")
    if args.out_html:
        out_html = Path(args.out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(generate_html_report(md), encoding="utf-8")
        print(f"[DONE] wrote {out_html}")


if __name__ == "__main__":
    main()
