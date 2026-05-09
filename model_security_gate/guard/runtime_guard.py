from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.detox.external_hard_suite import apply_overlap_class_guard_to_detections
from model_security_gate.guard.semantic_shortcut_guard import (
    SemanticShortcutGuardConfig,
    decide_semantic_shortcut_guard,
)
from model_security_gate.scan.tta_scan import TTAScanConfig, run_tta_scan


@dataclass
class RuntimeGuardConfig:
    context_drop_high: float = 0.60
    target_removal_conf_high: float = 0.50
    max_suspicious_rows: int = 1
    enable_semantic_shortcut_guard: bool = True
    semantic_shortcut_guard: SemanticShortcutGuardConfig | None = None
    enable_overlap_class_guard: bool = True
    overlap_guard_iou: float = 0.10
    overlap_guard_conf_margin: float = 0.30
    overlap_guard_min_suppressor_conf: float = 0.25


def _default_suppressor_class_ids(adapter: ModelAdapter, target_class_ids: Sequence[int]) -> list[int]:
    names = {int(k): str(v).lower() for k, v in adapter.names.items()}
    target_names = {names.get(int(cls_id), str(cls_id)).lower() for cls_id in target_class_ids}
    suppressors: list[int] = []
    if "helmet" in target_names:
        suppressors.extend([cls_id for cls_id, name in names.items() if name == "head"])
    return sorted(set(suppressors))


def guard_image(
    adapter: ModelAdapter,
    image_path: str | Path,
    target_class_ids: Sequence[int],
    cfg: RuntimeGuardConfig | None = None,
    tta_cfg: TTAScanConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or RuntimeGuardConfig()
    tta_cfg = tta_cfg or TTAScanConfig(
        context_drop_high=cfg.context_drop_high,
        target_removal_conf_high=cfg.target_removal_conf_high,
    )
    generator = CounterfactualGenerator(
        variants=["grayscale", "low_saturation", "hue_rotate", "jpeg", "blur", "context_occlude", "target_occlude"]
    )
    df = run_tta_scan(adapter, [image_path], target_class_ids=target_class_ids, generator=generator, cfg=tta_cfg)
    base_detections = adapter.predict_image(image_path)
    served_detections: list[Detection] = list(base_detections)
    overlap_guard = {"action": "pass", "matched_rules": [], "removed_detections": 0}
    if cfg.enable_overlap_class_guard:
        suppressor_ids = _default_suppressor_class_ids(adapter, target_class_ids)
        if suppressor_ids:
            served_detections, overlap_guard = apply_overlap_class_guard_to_detections(
                served_detections,
                target_class_ids,
                suppressor_class_ids=suppressor_ids,
                iou_threshold=cfg.overlap_guard_iou,
                conf_margin=cfg.overlap_guard_conf_margin,
                min_suppressor_conf=cfg.overlap_guard_min_suppressor_conf,
            )
    semantic_shortcut = (
        decide_semantic_shortcut_guard(
            image_path,
            served_detections,
            target_class_ids,
            cfg=cfg.semantic_shortcut_guard,
        )
        if cfg.enable_semantic_shortcut_guard
        else {"action": "pass", "matches": []}
    )
    if df.empty:
        if semantic_shortcut.get("action") == "review":
            return {
                "image": str(image_path),
                "verdict": "review",
                "reason": "semantic shortcut guard",
                "n_suspicious": int(len(semantic_shortcut.get("matches", []))),
                "rows": semantic_shortcut.get("matches", []),
                "n_served_target_detections": int(sum(1 for det in served_detections if det.cls_id in set(target_class_ids))),
                "overlap_guard": overlap_guard,
                "semantic_shortcut": semantic_shortcut,
            }
        return {
            "image": str(image_path),
            "verdict": "pass",
            "reason": "no target detections",
            "n_suspicious": 0,
            "rows": [],
            "n_served_target_detections": int(sum(1 for det in served_detections if det.cls_id in set(target_class_ids))),
            "overlap_guard": overlap_guard,
            "semantic_shortcut": semantic_shortcut,
        }
    suspicious = df[df.get("context_dependence", False).fillna(False) | df.get("target_removal_failure", False).fillna(False)]
    semantic_review = semantic_shortcut.get("action") == "review"
    verdict = "review" if len(suspicious) >= cfg.max_suspicious_rows or semantic_review else "pass"
    rows = suspicious.head(20).to_dict(orient="records")
    if semantic_review:
        rows.extend(semantic_shortcut.get("matches", []))
    return {
        "image": str(image_path),
        "verdict": verdict,
        "reason": "semantic shortcut guard" if semantic_review else ("counterfactual instability" if verdict == "review" else "stable enough"),
        "n_suspicious": int(len(rows)),
        "rows": rows,
        "n_served_target_detections": int(sum(1 for det in served_detections if det.cls_id in set(target_class_ids))),
        "overlap_guard": overlap_guard,
        "semantic_shortcut": semantic_shortcut,
    }


def guard_batch(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    target_class_ids: Sequence[int],
    output_csv: str | Path,
    cfg: RuntimeGuardConfig | None = None,
    tta_cfg: TTAScanConfig | None = None,
) -> dict:
    """Run the runtime guard on many images and write a compact CSV."""
    rows: List[Dict[str, Any]] = []
    for p in image_paths:
        result = guard_image(adapter, p, target_class_ids, cfg=cfg, tta_cfg=tta_cfg)
        rows.append(
            {
                "image": str(p),
                "image_basename": Path(p).name,
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "n_suspicious": int(result.get("n_suspicious", 0) or 0),
                "n_served_target_detections": int(result.get("n_served_target_detections", 0) or 0),
                "n_auto_target_detections": 0
                if result.get("verdict") == "review"
                else int(result.get("n_served_target_detections", 0) or 0),
                "suspicious_rows_json": json.dumps(result.get("rows", []), ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    n_review = int((df["verdict"] == "review").sum()) if not df.empty else 0
    return {
        "n_images": int(len(df)),
        "n_review": n_review,
        "review_rate": float(n_review / max(1, len(df))),
        "n_auto_target_detections": int(df["n_auto_target_detections"].sum()) if "n_auto_target_detections" in df else 0,
        "output_csv": str(output_csv),
    }
