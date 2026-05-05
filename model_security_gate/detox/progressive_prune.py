from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.detox.prune import save_ultralytics_model, zero_out_ranked_channels
from model_security_gate.scan.tta_scan import TTAScanConfig, run_tta_scan, summarize_tta
from model_security_gate.utils.io import write_json


@dataclass
class ProgressivePruneConfig:
    top_ks: Sequence[int] = (10, 25, 50, 100)
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_eval_images: int = 80
    select_metric: str = "target_removal_failure_rate"


def make_pruned_candidate(
    model_path: str | Path,
    ranked_channels: pd.DataFrame,
    output_path: str | Path,
    top_k: int,
    device: str | int | None = None,
) -> Path:
    adapter = UltralyticsYOLOAdapter(model_path, device=device)
    zero_out_ranked_channels(adapter, ranked_channels, top_k=int(top_k))
    return save_ultralytics_model(adapter, output_path)


def progressive_prune_and_select(
    model_path: str | Path,
    ranked_channels: pd.DataFrame,
    image_paths: Sequence[str | Path],
    labels_dir: str | Path | None,
    target_class_ids: Sequence[int],
    output_dir: str | Path,
    cfg: ProgressivePruneConfig | None = None,
    device: str | int | None = None,
) -> Dict[str, Any]:
    """Create several soft-pruned candidates and pick the best by TTA risk proxy."""
    cfg = cfg or ProgressivePruneConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(image_paths)[: cfg.max_eval_images]
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for top_k in cfg.top_ks:
        cand_path = output_dir / f"pruned_top_{int(top_k)}.pt"
        make_pruned_candidate(model_path, ranked_channels, cand_path, int(top_k), device=device)
        adapter = UltralyticsYOLOAdapter(cand_path, device=device, default_conf=cfg.conf, default_iou=cfg.iou, default_imgsz=cfg.imgsz)
        generator = CounterfactualGenerator()
        df = run_tta_scan(
            adapter,
            image_paths,
            labels_dir=labels_dir,
            target_class_ids=target_class_ids,
            generator=generator,
            cfg=TTAScanConfig(conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz),
        )
        summary = summarize_tta(df)
        metric = float(summary.get(cfg.select_metric, 0.0))
        # Tie-break: prefer less pruning when metric is equal.
        rec = {"top_k": int(top_k), "path": str(cand_path), "metric": metric, "tta_summary": summary}
        rows.append(rec)
        if best is None or (metric, int(top_k)) < (float(best["metric"]), int(best["top_k"])):
            best = rec

    manifest = {"config": asdict(cfg), "candidates": rows, "selected": best}
    write_json(output_dir / "progressive_prune_manifest.json", manifest)
    return manifest
