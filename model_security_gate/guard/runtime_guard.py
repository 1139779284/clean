from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from model_security_gate.adapters.base import ModelAdapter
from model_security_gate.cf.transforms import CounterfactualGenerator
from model_security_gate.scan.tta_scan import TTAScanConfig, run_tta_scan


@dataclass
class RuntimeGuardConfig:
    context_drop_high: float = 0.60
    target_removal_conf_high: float = 0.50
    max_suspicious_rows: int = 1


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
    if df.empty:
        return {"image": str(image_path), "verdict": "pass", "reason": "no target detections", "n_suspicious": 0, "rows": []}
    suspicious = df[df.get("context_dependence", False).fillna(False) | df.get("target_removal_failure", False).fillna(False)]
    verdict = "review" if len(suspicious) >= cfg.max_suspicious_rows else "pass"
    return {
        "image": str(image_path),
        "verdict": verdict,
        "reason": "counterfactual instability" if verdict == "review" else "stable enough",
        "n_suspicious": int(len(suspicious)),
        "rows": suspicious.head(20).to_dict(orient="records"),
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
        "output_csv": str(output_csv),
    }
