from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_security_gate.adapters.base import ModelAdapter


@dataclass
class ChannelScanConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_layers: int = 12
    max_channels_per_layer: int = 256
    layer_name_contains: Sequence[str] | None = None


@dataclass
class ChannelScanEvaluationConfig:
    min_images: int = 30
    min_rows: int = 20
    high_score_norm: float = 0.85
    high_abs_corr: float = 0.50
    high_positive_jump_rate: float = 0.30
    top_score_gap_warn: float = 0.35


def _get_torch_model(adapter: ModelAdapter):
    # UltralyticsYOLOAdapter: adapter.model is YOLO wrapper; adapter.model.model is nn.Module.
    if hasattr(adapter, "model") and hasattr(adapter.model, "model"):
        return adapter.model.model
    if hasattr(adapter, "torch_model"):
        return adapter.torch_model
    raise TypeError("Adapter does not expose an underlying torch model")


def _candidate_conv_modules(torch_model, cfg: ChannelScanConfig):
    mods = []
    contains = list(cfg.layer_name_contains or [])
    for name, mod in torch_model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            if contains and not any(c in name for c in contains):
                continue
            mods.append((name, mod))
    # Later layers are usually more class-specific; sample from the back if too many.
    if len(mods) > cfg.max_layers:
        idx = np.linspace(0, len(mods) - 1, cfg.max_layers).round().astype(int)
        mods = [mods[i] for i in idx]
    return mods


def _max_target_conf(adapter: ModelAdapter, path: str | Path, target_class_ids: Sequence[int], cfg: ChannelScanConfig) -> float:
    dets = adapter.predict_image(path, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
    best = 0.0
    wanted = set(int(x) for x in target_class_ids)
    for d in dets:
        if d.cls_id in wanted:
            best = max(best, float(d.conf))
    return best


def run_channel_correlation_scan(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    target_class_ids: Sequence[int],
    cfg: ChannelScanConfig | None = None,
) -> pd.DataFrame:
    """Rank channels whose activation correlates with target-class confidence.

    This is not a proof of a backdoor. It is a practical triage tool: channels
    that activate rarely on clean data but correlate strongly with a critical
    class are candidates for manual review or conservative soft-ablation.
    """
    cfg = cfg or ChannelScanConfig()
    torch_model = _get_torch_model(adapter)
    modules = _candidate_conv_modules(torch_model, cfg)
    if not modules:
        return pd.DataFrame()

    # Store per-image channel means: module_name -> list[np.ndarray]
    accum: Dict[str, List[np.ndarray]] = {name: [] for name, _ in modules}
    hooks = []

    def make_hook(name):
        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)):
                out_t = out[0]
            else:
                out_t = out
            if not torch.is_tensor(out_t) or out_t.ndim < 4:
                return
            # mean absolute activation per channel over batch/spatial.
            val = out_t.detach().abs().mean(dim=(0, 2, 3)).float().cpu().numpy()
            if val.shape[0] > cfg.max_channels_per_layer:
                # keep evenly-spaced sample indices; pruning can still use these exact indices.
                idx = np.linspace(0, val.shape[0] - 1, cfg.max_channels_per_layer).round().astype(int)
                val2 = np.zeros_like(val)
                val2[idx] = val[idx]
                val = val2
            accum[name].append(val)
        return hook

    for name, mod in modules:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    target_confs: List[float] = []
    try:
        for path in tqdm(list(image_paths), desc="Channel correlation scan"):
            target_confs.append(_max_target_conf(adapter, path, target_class_ids, cfg))
    finally:
        for h in hooks:
            h.remove()

    y = np.asarray(target_confs, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    if len(y) < 3 or np.std(y) < 1e-6:
        # Still report activation rarity if target confidence has no variation.
        for name, vals in accum.items():
            if not vals:
                continue
            arr = np.vstack(vals)
            mean = arr.mean(axis=0)
            p95 = np.percentile(arr, 95, axis=0)
            for ch in np.nonzero(mean)[0]:
                rows.append({"module": name, "channel": int(ch), "corr_with_target_conf": 0.0, "mean_abs_activation": float(mean[ch]), "p95_abs_activation": float(p95[ch]), "score": float(p95[ch])})
        return pd.DataFrame(rows).sort_values("score", ascending=False)

    yz = (y - y.mean()) / (y.std() + 1e-6)
    for name, vals in accum.items():
        if not vals:
            continue
        arr = np.vstack(vals).astype(np.float32)
        mean = arr.mean(axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        std = arr.std(axis=0) + 1e-6
        xz = (arr - mean) / std
        corr = (xz * yz[:, None]).mean(axis=0)
        rarity = p95 / (mean + 1e-4)
        score = np.abs(corr) * np.log1p(p95) * np.log1p(rarity)
        for ch in np.argsort(score)[-min(30, len(score)) :]:
            rows.append(
                {
                    "module": name,
                    "channel": int(ch),
                    "corr_with_target_conf": float(corr[ch]),
                    "mean_abs_activation": float(mean[ch]),
                    "p95_abs_activation": float(p95[ch]),
                    "rarity_ratio": float(rarity[ch]),
                    "score": float(score[ch]),
                }
            )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_channel_scan(
    df: pd.DataFrame,
    n_images: int | None = None,
    cfg: ChannelScanEvaluationConfig | None = None,
) -> Dict[str, Any]:
    """Convert channel rankings into a conservative, reproducible quality gate.

    Channel evidence is useful for pruning candidates, but it is noisy unless
    enough images were scanned and the top-ranked channels are stable outliers.
    This evaluator makes that limitation explicit in reports instead of treating
    every channel CSV as equally strong evidence.
    """
    cfg = cfg or ChannelScanEvaluationConfig()
    n_rows = int(0 if df is None else len(df))
    if df is None or df.empty:
        return {
            "status": "not_available",
            "evidence_strength": "none",
            "n_images": int(n_images or 0),
            "n_rows": 0,
            "high_risk_channels": 0,
            "warnings": ["channel scan produced no rows"],
        }

    warnings: List[str] = []
    if n_images is None:
        warnings.append("n_images not provided; stability cannot be fully assessed")
    elif int(n_images) < cfg.min_images:
        warnings.append(f"only {int(n_images)} images scanned; recommended >= {cfg.min_images}")
    if n_rows < cfg.min_rows:
        warnings.append(f"only {n_rows} channel rows scored; recommended >= {cfg.min_rows}")

    def numeric_column(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)

    score_col = "detox_score" if "detox_score" in df.columns else "score"
    scores = numeric_column(score_col)
    if len(scores) and float(scores.max()) > float(scores.min()):
        score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        score_norm = pd.Series(np.zeros(len(scores)), index=df.index)

    abs_corr = numeric_column("corr_with_target_conf").abs()
    jump_rate = numeric_column("positive_jump_rate")
    has_anp = "anp_score" in df.columns or "positive_jump_rate" in df.columns

    high_mask = (score_norm >= cfg.high_score_norm) | (abs_corr >= cfg.high_abs_corr) | (jump_rate >= cfg.high_positive_jump_rate)
    high_risk_channels = int(high_mask.sum())

    sorted_scores = score_norm.sort_values(ascending=False).to_numpy()
    top_score_norm = float(sorted_scores[0]) if len(sorted_scores) else 0.0
    second_score_norm = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    top_score_gap = top_score_norm - second_score_norm
    if top_score_gap >= cfg.top_score_gap_warn:
        warnings.append("top channel is a strong isolated outlier; review before pruning")

    stability_ready = bool((n_images is None or int(n_images) >= cfg.min_images) and n_rows >= cfg.min_rows)
    if not stability_ready:
        status = "insufficient_data"
    elif high_risk_channels:
        status = "review"
    else:
        status = "pass"

    if not stability_ready:
        evidence_strength = "weak"
    elif has_anp:
        evidence_strength = "strong" if high_risk_channels else "medium"
    else:
        evidence_strength = "medium" if high_risk_channels else "weak"

    return {
        "status": status,
        "evidence_strength": evidence_strength,
        "n_images": int(n_images or 0),
        "n_rows": n_rows,
        "score_column": score_col,
        "top_score_norm": top_score_norm,
        "top_score_gap": float(top_score_gap),
        "high_risk_channels": high_risk_channels,
        "has_anp_evidence": bool(has_anp),
        "warnings": warnings,
        "criteria": {
            "min_images": cfg.min_images,
            "min_rows": cfg.min_rows,
            "high_score_norm": cfg.high_score_norm,
            "high_abs_corr": cfg.high_abs_corr,
            "high_positive_jump_rate": cfg.high_positive_jump_rate,
        },
    }


def summarize_channel_scan(df: pd.DataFrame, top_k: int = 20, n_images: int | None = None) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "top_channels": [], "evaluation": evaluate_channel_scan(df, n_images=n_images)}
    return {
        "n_rows": int(len(df)),
        "top_channels": df.head(top_k).to_dict(orient="records"),
        "evaluation": evaluate_channel_scan(df, n_images=n_images),
    }
