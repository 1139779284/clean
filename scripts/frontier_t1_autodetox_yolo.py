#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.semantic_surgical import SemanticSurgicalRepairConfig, run_frontier_auto_semantic_detox


def _parse_name_value(values: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        out[name.strip()] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatic frontier T1 last-mile semantic detox profile ladder")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--external-roots", nargs="+", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--semantic-attack-names", nargs="*", default=None)
    p.add_argument("--guard-attack-names", nargs="*", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--letterbox-train", action="store_true")
    p.add_argument("--max-images-per-attack", type=int, default=20)
    p.add_argument("--replay-max-images-per-attack", type=int, default=20)
    p.add_argument("--semantic-failure-repeat", type=int, default=24)
    p.add_argument("--guard-repeat", type=int, default=2)
    p.add_argument("--semantic-fp-required-max-conf", type=float, default=0.25)
    p.add_argument("--max-attack-asr", nargs="*", default=None)
    p.add_argument("--max-single-attack-worsen", type=float, default=0.0)
    p.add_argument("--max-allowed-external-asr", type=float, default=0.05)
    p.add_argument("--level", default="last_mile", choices=["last_mile", "strong", "frontier", "aggressive"])
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SemanticSurgicalRepairConfig(
        model=args.model,
        data_yaml=args.data_yaml,
        out_dir=args.out,
        external_roots=tuple(args.external_roots),
        target_classes=tuple(args.target_classes),
        semantic_attack_names=tuple(args.semantic_attack_names or ()),
        guard_attack_names=tuple(args.guard_attack_names or ()),
        teacher_model=args.teacher_model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        letterbox_train=args.letterbox_train,
        max_images_per_attack=args.max_images_per_attack,
        replay_max_images_per_attack=args.replay_max_images_per_attack,
        semantic_failure_repeat=args.semantic_failure_repeat,
        guard_repeat=args.guard_repeat,
        max_attack_asr=_parse_name_value(args.max_attack_asr),
        semantic_fp_required_max_conf=args.semantic_fp_required_max_conf,
        max_single_attack_worsen=args.max_single_attack_worsen,
        max_allowed_external_asr=args.max_allowed_external_asr,
        amp=args.amp,
    )
    manifest = run_frontier_auto_semantic_detox(cfg, level=args.level)
    print(json.dumps({k: manifest.get(k) for k in ("status", "final_model", "selected_profile", "rolled_back", "best_any")}, indent=2, ensure_ascii=False))
    print(f"[DONE] manifest: {Path(args.out) / 'frontier_auto_semantic_detox_manifest.json'}")


if __name__ == "__main__":
    main()
