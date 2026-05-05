#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Dict, List

import pandas as pd

from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, write_resolved_config
from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.scan.occlusion_attribution import OcclusionConfig, run_occlusion_attribution_scan, summarize_occlusion
from model_security_gate.scan.risk import compute_risk_score, load_risk_config
from model_security_gate.scan.slice_scan import run_slice_scan, summarize_slice
from model_security_gate.scan.stress_suite import run_stress_suite, summarize_stress
from model_security_gate.scan.tta_scan import TTAScanConfig, run_tta_scan, summarize_tta
from model_security_gate.utils.io import list_images, resolve_class_ids, write_json


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_summary(model_path: Path) -> Dict[str, Any]:
    exists = model_path.exists()
    return {
        "model_path": str(model_path),
        "exists": exists,
        "sha256": sha256_file(model_path) if exists and model_path.is_file() else None,
        # Users can replace this with their own supply-chain policy.
        "risk": 0.4 if exists else 1.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-trust backdoor/security gate for YOLO detectors")
    p.add_argument("--config", default=None, help="YAML config. Values under `security_gate:` are also accepted. CLI args override YAML.")
    p.add_argument("--model", default=None, help="Path to YOLO .pt/.onnx supported by Ultralytics; .pt recommended")
    p.add_argument("--images", default=None, help="Image directory or single image")
    p.add_argument("--labels", default=None, help="YOLO labels directory. Optional but strongly recommended.")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--critical-classes", nargs="+", default=None, help="Class names or class IDs, e.g. helmet person")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--occlusion", action="store_true", default=None, help="Run slower black-box occlusion attribution")
    p.add_argument("--occlusion-max-images", type=int, default=None)
    p.add_argument("--channel-scan", action="store_true", default=None, help="Run experimental channel correlation scan on .pt models")
    p.add_argument("--channel-max-images", type=int, default=None)
    p.add_argument("--risk-config", default=None, help="Optional YAML with risk weights/thresholds")
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "model": None,
        "images": None,
        "labels": None,
        "out": "runs/security_gate",
        "critical_classes": None,
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "device": None,
        "max_images": 200,
        "occlusion": False,
        "occlusion_max_images": 20,
        "channel_scan": False,
        "channel_max_images": 80,
        "risk_config": None,
    }
    cfg = load_yaml_config(args.config, section="security_gate")
    resolved = deep_merge(defaults, deep_merge(cfg, namespace_overrides(args, exclude={"config"})))
    missing = [k for k in ["model", "images", "critical_classes"] if not resolved.get(k)]
    if missing:
        raise SystemExit(f"Missing required config/CLI values: {', '.join(missing)}")
    return argparse.Namespace(**resolved)


def main() -> None:
    args = resolve_args(parse_args())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(out / "resolved_config.json", vars(args))

    image_paths = list_images(args.images, max_images=args.max_images)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    adapter = UltralyticsYOLOAdapter(args.model, device=args.device, default_conf=args.conf, default_iou=args.iou, default_imgsz=args.imgsz)
    target_ids = resolve_class_ids(adapter.names, args.critical_classes)
    print(f"[INFO] Critical class IDs: {target_ids}; names={adapter.names}")

    summaries: Dict[str, Dict[str, Any]] = {"provenance": provenance_summary(Path(args.model))}

    # 1. Slice scan: detects attribute-conditioned false positives if labels exist.
    slice_df = run_slice_scan(
        adapter,
        image_paths,
        labels_dir=args.labels,
        target_class_ids=target_ids,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    slice_df.to_csv(out / "slice_scan.csv", index=False)
    summaries["slice"] = summarize_slice(slice_df)

    # 2. Counterfactual TTA scan: target/context occlusion + color/texture perturbation.
    generator = CounterfactualGenerator(
        variants=[
            "grayscale",
            "low_saturation",
            "hue_rotate",
            "brightness_contrast",
            "jpeg",
            "blur",
            "random_patch",
            "context_occlude",
            "target_occlude",
        ]
    )
    tta_cfg = TTAScanConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    tta_df = run_tta_scan(adapter, image_paths, labels_dir=args.labels, target_class_ids=target_ids, generator=generator, cfg=tta_cfg)
    tta_df.to_csv(out / "tta_scan.csv", index=False)
    summaries["tta"] = summarize_tta(tta_df)

    # 3. Unknown-trigger stress suite.
    stress_df = run_stress_suite(
        adapter,
        image_paths,
        labels_dir=args.labels,
        target_class_ids=target_ids,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    stress_df.to_csv(out / "stress_suite.csv", index=False)
    summaries["stress"] = summarize_stress(stress_df)

    # 4. Optional black-box occlusion attribution. Use high-risk images first.
    if args.occlusion:
        suspicious_images: List[str] = []
        if not tta_df.empty:
            suspicious_images = tta_df[
                tta_df.get("context_dependence", False).fillna(False) | tta_df.get("target_removal_failure", False).fillna(False)
            ]["image"].drop_duplicates().tolist()
        occ_paths = [Path(x) for x in suspicious_images[: args.occlusion_max_images]]
        if len(occ_paths) < args.occlusion_max_images:
            extra = [p for p in image_paths if p not in set(occ_paths)]
            occ_paths.extend(extra[: args.occlusion_max_images - len(occ_paths)])
        occ_cfg = OcclusionConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz, save_heatmaps=True)
        occ_df = run_occlusion_attribution_scan(adapter, occ_paths, target_class_ids=target_ids, output_dir=out / "heatmaps", cfg=occ_cfg)
        occ_df.to_csv(out / "occlusion_attribution.csv", index=False)
        summaries["occlusion"] = summarize_occlusion(occ_df)
    else:
        summaries["occlusion"] = {"n_rows": 0, "wrong_region_attention_rate": 0.0, "mean_mass_in_box": 0.0}

    # 5. Optional channel correlation scan. Treat as weak evidence and pruning hint.
    if args.channel_scan:
        from model_security_gate.scan.neuron_sensitivity import ChannelScanConfig, run_channel_correlation_scan, summarize_channel_scan

        ch_paths = image_paths[: args.channel_max_images]
        ch_df = run_channel_correlation_scan(adapter, ch_paths, target_class_ids=target_ids, cfg=ChannelScanConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz))
        ch_df.to_csv(out / "channel_scan.csv", index=False)
        summaries["channel"] = summarize_channel_scan(ch_df)
    else:
        summaries["channel"] = {"n_rows": 0, "top_channels": []}

    risk_weights, risk_thresholds = load_risk_config(args.risk_config)
    decision = compute_risk_score(summaries, weights=risk_weights, thresholds=risk_thresholds)
    report = {
        "decision": decision,
        "summaries": summaries,
        "critical_class_ids": target_ids,
        "critical_classes": [adapter.names.get(i, str(i)) for i in target_ids],
        "n_images": len(image_paths),
        "outputs": {
            "slice_scan": str(out / "slice_scan.csv"),
            "tta_scan": str(out / "tta_scan.csv"),
            "stress_suite": str(out / "stress_suite.csv"),
            "occlusion_attribution": str(out / "occlusion_attribution.csv") if args.occlusion else None,
            "channel_scan": str(out / "channel_scan.csv") if args.channel_scan else None,
        },
    }
    write_json(out / "security_report.json", report)
    print(f"[DONE] level={decision.level} score={decision.score}")
    for r in decision.reasons:
        print(f"  - {r}")
    print(f"[DONE] report: {out / 'security_report.json'}")


if __name__ == "__main__":
    main()
