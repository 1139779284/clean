from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_security_gate.detox.feature_hooks import ActivationCatcher, select_conv_layers
from model_security_gate.detox.yolo_dataset import move_batch_to_device


@dataclass
class FMPScoreConfig:
    max_batches: int = 50
    max_layers: int = 10
    max_channels_per_layer: int = 512
    layer_name_contains: Sequence[str] | None = None
    clean_percentile: float = 50.0
    hard_percentile: float = 95.0


def _collect_channel_stats(
    model: torch.nn.Module,
    loader,
    layer_names: Sequence[str],
    max_batches: int,
    device: torch.device,
    desc: str,
) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {name: [] for name in layer_names}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=desc)):
            if bi >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            with ActivationCatcher(model, layer_names) as ac:
                _ = model(batch["img"])
            for name, feat in ac.features.items():
                if feat.ndim == 4:
                    vals = feat.detach().abs().mean(dim=(0, 2, 3)).float().cpu().numpy()
                    out[name].append(vals)
    model.train(was_training)
    return out


def compute_fmp_channel_scores(
    model: torch.nn.Module,
    clean_loader,
    hard_loader=None,
    cfg: FMPScoreConfig | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Feature-map pruning score.

    FMP intuition: channels that are dormant or weak on clean data but spike on
    suspicious/counterfactual hard data can encode backdoor information. If no
    separate hard_loader is provided, this still returns clean rarity scores.
    """
    cfg = cfg or FMPScoreConfig()
    device = torch.device(device)
    layer_names = select_conv_layers(model, contains=cfg.layer_name_contains, max_layers=cfg.max_layers, prefer_late=True)
    if not layer_names:
        return pd.DataFrame()
    clean_stats = _collect_channel_stats(model, clean_loader, layer_names, cfg.max_batches, device, "FMP clean stats")
    hard_stats = _collect_channel_stats(model, hard_loader, layer_names, cfg.max_batches, device, "FMP hard stats") if hard_loader is not None else {}

    rows: List[Dict[str, Any]] = []
    for name in layer_names:
        vals = clean_stats.get(name, [])
        if not vals:
            continue
        c_arr = np.vstack(vals)
        c_mean = c_arr.mean(axis=0)
        c_med = np.percentile(c_arr, cfg.clean_percentile, axis=0)
        c_p95 = np.percentile(c_arr, 95, axis=0)
        if name in hard_stats and hard_stats[name]:
            h_arr = np.vstack(hard_stats[name])
            h_p = np.percentile(h_arr, cfg.hard_percentile, axis=0)
            spike = h_p / (c_med + 1e-5)
        else:
            h_p = c_p95
            spike = c_p95 / (c_mean + 1e-5)
        dormancy = 1.0 / (c_mean + 1e-5)
        score = np.log1p(spike) * np.log1p(dormancy) * np.log1p(h_p)
        keep = np.argsort(score)[-min(len(score), cfg.max_channels_per_layer):]
        for ch in keep:
            rows.append(
                {
                    "module": name,
                    "channel": int(ch),
                    "fmp_score": float(score[ch]),
                    "score": float(score[ch]),
                    "clean_mean_activation": float(c_mean[ch]),
                    "clean_p95_activation": float(c_p95[ch]),
                    "hard_p95_activation": float(h_p[ch]),
                    "hard_clean_spike": float(spike[ch]),
                    "method": "feature_map_pruning",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
