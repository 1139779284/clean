#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Score suspicious channels for strong detox pruning")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--target-classes", nargs="+", required=True)
    p.add_argument("--out", default="runs/detox_channel_scores.csv")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=120)
    p.add_argument("--no-anp", action="store_true")
    p.add_argument("--anp-max-images", type=int, default=32)
    p.add_argument("--summary-out", default=None, help="Optional JSON summary with channel evidence stability gate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Heavy imports are intentionally lazy so `--help` stays fast and does not
    # initialize torch/Ultralytics.
    from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
    from model_security_gate.detox.channel_scoring import ANPSensitivityConfig, score_channels_for_detox
    from model_security_gate.scan.neuron_sensitivity import ChannelScanConfig, summarize_channel_scan
    from model_security_gate.utils.io import list_images, load_class_names_from_data_yaml, resolve_class_ids
    from model_security_gate.utils.io import write_json

    names = load_class_names_from_data_yaml(args.data_yaml)
    adapter = UltralyticsYOLOAdapter(args.model, device=args.device, default_conf=args.conf, default_iou=args.iou, default_imgsz=args.imgsz)
    if not names:
        names = adapter.names
    target_ids = resolve_class_ids(names, args.target_classes)
    paths = list_images(args.images, max_images=args.max_images)
    df = score_channels_for_detox(
        adapter,
        paths,
        target_ids,
        corr_cfg=ChannelScanConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz),
        anp_cfg=ANPSensitivityConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz, max_images=args.anp_max_images),
        run_anp=not args.no_anp,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    write_json(summary_out, summarize_channel_scan(df, n_images=len(paths)))
    print(f"[DONE] wrote {out} rows={len(df)} target_ids={target_ids}")
    print(f"[DONE] summary: {summary_out}")


if __name__ == "__main__":
    main()
