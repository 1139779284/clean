from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_security_gate.adapters.base import ModelAdapter
from model_security_gate.scan.neuron_sensitivity import ChannelScanConfig, run_channel_correlation_scan
from model_security_gate.detox.common import get_torch_model, list_conv_modules, select_evenly


@dataclass
class ANPSensitivityConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_images: int = 32
    max_layers: int = 8
    max_channels_per_layer: int = 32
    channel_gain: float = 1.5
    layer_name_contains: Sequence[str] | None = None


def _max_target_conf(adapter: ModelAdapter, path: str | Path, target_class_ids: Sequence[int], cfg: ANPSensitivityConfig) -> float:
    wanted = set(int(x) for x in target_class_ids)
    best = 0.0
    for d in adapter.predict_image(path, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz):
        if d.cls_id in wanted:
            best = max(best, float(d.conf))
    return best


def run_anp_channel_sensitivity_scan(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    target_class_ids: Sequence[int],
    cfg: ANPSensitivityConfig | None = None,
) -> pd.DataFrame:
    """ANP-style channel sensitivity scan by temporarily amplifying channels.

    This is deliberately conservative and slow: it samples layers/channels,
    amplifies one channel at a time through a forward hook, and measures whether
    critical-class confidence jumps. Channels with high positive jumps are strong
    candidates for soft ablation or progressive pruning.
    """
    cfg = cfg or ANPSensitivityConfig()
    image_paths = list(image_paths)[: cfg.max_images]
    if not image_paths or not target_class_ids:
        return pd.DataFrame()

    torch_model = get_torch_model(adapter)
    convs = select_evenly(list_conv_modules(torch_model, cfg.layer_name_contains), cfg.max_layers)
    base = np.asarray([_max_target_conf(adapter, p, target_class_ids, cfg) for p in tqdm(image_paths, desc="ANP baseline")], dtype=np.float32)
    rows: List[Dict[str, Any]] = []

    for module_name, module in tqdm(convs, desc="ANP layers"):
        n_ch = int(module.out_channels)
        channels = select_evenly(list(range(n_ch)), min(cfg.max_channels_per_layer, n_ch))
        for ch in channels:
            def hook(_mod, _inp, out, ch=ch):
                x = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(x) or x.ndim != 4 or ch >= x.shape[1]:
                    return out
                y = x.clone()
                y[:, ch, :, :] = y[:, ch, :, :] * float(cfg.channel_gain)
                return y

            handle = module.register_forward_hook(hook)
            try:
                pert = np.asarray([_max_target_conf(adapter, p, target_class_ids, cfg) for p in image_paths], dtype=np.float32)
            finally:
                handle.remove()
            delta = pert - base
            rows.append(
                {
                    "module": module_name,
                    "channel": int(ch),
                    "base_mean_target_conf": float(base.mean()) if len(base) else 0.0,
                    "pert_mean_target_conf": float(pert.mean()) if len(pert) else 0.0,
                    "mean_conf_delta": float(delta.mean()) if len(delta) else 0.0,
                    "p95_conf_delta": float(np.percentile(delta, 95)) if len(delta) else 0.0,
                    "positive_jump_rate": float((delta > 0.05).mean()) if len(delta) else 0.0,
                    "anp_score": float(max(0.0, delta.mean()) + max(0.0, np.percentile(delta, 95)) + 0.5 * (delta > 0.05).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("anp_score", ascending=False).reset_index(drop=True)


def merge_channel_evidence(correlation_df: pd.DataFrame, anp_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge correlation, rarity and ANP sensitivity evidence into one ranking."""
    if correlation_df is None or correlation_df.empty:
        base = pd.DataFrame(columns=["module", "channel"])
    else:
        base = correlation_df.copy()
    if "score" not in base.columns:
        base["score"] = 0.0
    base["module"] = base.get("module", pd.Series(dtype=str)).astype(str)
    base["channel"] = pd.to_numeric(base.get("channel", pd.Series(dtype=float)), errors="coerce").fillna(-1).astype(int)

    if anp_df is not None and not anp_df.empty:
        anp = anp_df.copy()
        anp["module"] = anp["module"].astype(str)
        anp["channel"] = pd.to_numeric(anp["channel"], errors="coerce").fillna(-1).astype(int)
        merged = pd.merge(base, anp, on=["module", "channel"], how="outer", suffixes=("_corr", "_anp"))
    else:
        merged = base.copy()
        merged["anp_score"] = 0.0

    corr_score = pd.to_numeric(merged.get("score", 0.0), errors="coerce").fillna(0.0)
    anp_score = pd.to_numeric(merged.get("anp_score", 0.0), errors="coerce").fillna(0.0)
    rarity = pd.to_numeric(merged.get("rarity_ratio", 0.0), errors="coerce").fillna(0.0)

    def norm(s: pd.Series) -> pd.Series:
        if len(s) == 0 or float(s.max()) <= float(s.min()):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    merged["corr_score_norm"] = norm(corr_score)
    merged["anp_score_norm"] = norm(anp_score)
    merged["rarity_norm"] = norm(np.log1p(rarity))
    merged["detox_score"] = 0.45 * merged["corr_score_norm"] + 0.40 * merged["anp_score_norm"] + 0.15 * merged["rarity_norm"]
    return merged.sort_values("detox_score", ascending=False).reset_index(drop=True)


def score_channels_for_detox(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    target_class_ids: Sequence[int],
    corr_cfg: ChannelScanConfig | None = None,
    anp_cfg: ANPSensitivityConfig | None = None,
    run_anp: bool = True,
) -> pd.DataFrame:
    corr = run_channel_correlation_scan(adapter, image_paths, target_class_ids, cfg=corr_cfg or ChannelScanConfig())
    anp = run_anp_channel_sensitivity_scan(adapter, image_paths, target_class_ids, cfg=anp_cfg or ANPSensitivityConfig()) if run_anp else None
    return merge_channel_evidence(corr, anp)
