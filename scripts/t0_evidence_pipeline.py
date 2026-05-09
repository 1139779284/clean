#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or plan the full T0 evidence pipeline: audit, guard-free, trigger-only, guarded, clean mAP, evidence pack."
    )
    parser.add_argument("--model", required=True, help="Purified candidate model")
    parser.add_argument("--poisoned-model", default=None, help="Pre-detox poisoned model for clean-before metrics")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--benchmark-root", required=True, help="Corrected full benchmark root")
    parser.add_argument("--trigger-only-root", default=None, help="Trigger-only filtered benchmark root")
    parser.add_argument("--heldout-roots", nargs="*", default=[])
    parser.add_argument("--target-classes", nargs="+", default=["helmet"])
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--out", default="runs/t0_evidence_pipeline")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--apply-overlap-class-guard", action="store_true", default=True)
    parser.add_argument("--no-overlap-class-guard", action="store_false", dest="apply_overlap_class_guard")
    parser.add_argument("--overlap-guard-suppressor-class-ids", nargs="*", default=["1"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Without this, only writes the plan.")
    return parser.parse_args()


def _python() -> str:
    return sys.executable


def _append_if(values: list[str], flag: str, items: list[str] | tuple[str, ...]) -> None:
    if items:
        values.append(flag)
        values.extend(str(x) for x in items)


def build_pipeline_commands(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, str]]:
    out = Path(args.out)
    paths = {
        "benchmark_audit": str(out / "benchmark_audit" / "t0_benchmark_audit.json"),
        "guard_free_external": str(out / "guard_free_external" / "external_hard_suite_asr.json"),
        "guarded_external": str(out / "guarded_external" / "external_hard_suite_asr.json"),
        "trigger_only_external": str(out / "trigger_only_external" / "external_hard_suite_asr.json"),
        "clean_before": str(out / "clean_before_metrics.json"),
        "clean_after": str(out / "clean_after_metrics.json"),
        "evidence_pack": str(out / "evidence_pack" / "t0_evidence_pack.json"),
    }

    audit = [
        _python(),
        "scripts/t0_benchmark_audit.py",
        "--roots",
        args.benchmark_root,
        "--target-class-id",
        str(args.target_class_id),
        "--out",
        paths["benchmark_audit"],
    ]
    _append_if(audit, "--heldout-roots", list(args.heldout_roots))

    def external_cmd(root: str, dest: str, *, guarded: bool = False) -> list[str]:
        cmd = [
            _python(),
            "scripts/run_external_hard_suite.py",
            "--model",
            args.model,
            "--data-yaml",
            args.data_yaml,
            "--target-classes",
            *args.target_classes,
            "--roots",
            root,
            "--out",
            str(Path(dest).parent),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--match-iou",
            str(args.match_iou),
            "--oda-success-mode",
            "localized_any_recalled",
            "--device",
            str(args.device),
        ]
        if guarded and args.apply_overlap_class_guard:
            cmd.append("--apply-overlap-class-guard")
            _append_if(cmd, "--overlap-guard-suppressor-class-ids", list(args.overlap_guard_suppressor_class_ids))
        return cmd

    commands: list[dict[str, object]] = [
        {"name": "benchmark_audit", "output": paths["benchmark_audit"], "cmd": audit},
        {
            "name": "guard_free_external",
            "output": paths["guard_free_external"],
            "cmd": external_cmd(args.benchmark_root, paths["guard_free_external"]),
        },
        {
            "name": "guarded_external",
            "output": paths["guarded_external"],
            "cmd": external_cmd(args.benchmark_root, paths["guarded_external"], guarded=True),
        },
    ]

    if args.trigger_only_root:
        commands.append(
            {
                "name": "trigger_only_external",
                "output": paths["trigger_only_external"],
                "cmd": external_cmd(args.trigger_only_root, paths["trigger_only_external"]),
            }
        )

    if args.poisoned_model:
        commands.append(
            {
                "name": "clean_before",
                "output": paths["clean_before"],
                "cmd": [
                    _python(),
                    "scripts/eval_yolo_metrics.py",
                    "--model",
                    args.poisoned_model,
                    "--data-yaml",
                    args.data_yaml,
                    "--out",
                    paths["clean_before"],
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--workers",
                    str(args.workers),
                    "--device",
                    str(args.device),
                ],
            }
        )

    commands.append(
        {
            "name": "clean_after",
            "output": paths["clean_after"],
            "cmd": [
                _python(),
                "scripts/eval_yolo_metrics.py",
                "--model",
                args.model,
                "--data-yaml",
                args.data_yaml,
                "--out",
                paths["clean_after"],
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
                "--device",
                str(args.device),
            ],
        }
    )

    evidence_cmd = [
        _python(),
        "scripts/t0_evidence_pack.py",
        "--guard-free-external",
        paths["guard_free_external"],
        "--guarded-external",
        paths["guarded_external"],
        "--clean-after",
        paths["clean_after"],
        "--benchmark-audit",
        paths["benchmark_audit"],
        "--out",
        str(Path(paths["evidence_pack"]).parent),
    ]
    if args.trigger_only_root:
        evidence_cmd.extend(["--trigger-only-external", paths["trigger_only_external"]])
    if args.poisoned_model:
        evidence_cmd.extend(["--clean-before", paths["clean_before"]])
    commands.append({"name": "evidence_pack", "output": paths["evidence_pack"], "cmd": evidence_cmd})
    return commands, paths


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commands, paths = build_pipeline_commands(args)
    manifest = {"executed": bool(args.execute), "paths": paths, "commands": commands}
    (out / "t0_evidence_pipeline_manifest.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.execute:
        print(f"[PLAN] wrote {out / 't0_evidence_pipeline_manifest.json'}")
        return 0
    for row in commands:
        output = Path(str(row["output"]))
        if args.skip_existing and output.exists():
            print(f"[SKIP] {row['name']} -> {output}")
            continue
        print(f"[RUN] {row['name']}")
        subprocess.run([str(x) for x in row["cmd"]], check=True)
    print(f"[DONE] evidence: {Path(paths['evidence_pack']).parent / 'T0_EVIDENCE_PACK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
