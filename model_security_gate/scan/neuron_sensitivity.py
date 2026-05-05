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


def summarize_channel_scan(df: pd.DataFrame, top_k: int = 20) -> Dict[str, Any]:
    if df.empty:
        return {"n_rows": 0, "top_channels": []}
    return {"n_rows": int(len(df)), "top_channels": df.head(top_k).to_dict(orient="records")}
