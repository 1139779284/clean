from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_security_gate.detox.feature_hooks import ActivationCatcher, select_conv_layers
from model_security_gate.detox.losses import supervised_yolo_loss
from model_security_gate.detox.yolo_dataset import move_batch_to_device


@dataclass
class ANPScoreConfig:
    max_batches: int = 50
    max_layers: int = 10
    max_channels_per_layer: int = 512
    layer_name_contains: Sequence[str] | None = None
    loss_mode: str = "supervised"


def compute_anp_channel_scores(
    model: torch.nn.Module,
    loader,
    cfg: ANPScoreConfig | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Rank channels by ANP-style adversarial-neuron sensitivity.

    The original ANP learns adversarial neuron perturbations. This practical
    detector uses the same observation behind ANP: backdoor-related channels
    tend to be unusually loss-sensitive. For each captured feature map, it
    accumulates |activation * gradient| and |gradient| per channel under the
    supervised detox loss. High-ranking channels are pruning candidates.
    """
    cfg = cfg or ANPScoreConfig()
    device = torch.device(device)
    layer_names = select_conv_layers(
        model,
        contains=cfg.layer_name_contains,
        max_layers=cfg.max_layers,
        prefer_late=True,
    )
    if not layer_names:
        return pd.DataFrame()

    sums: Dict[str, Dict[str, torch.Tensor]] = {}
    counts: Dict[str, int] = {name: 0 for name in layer_names}
    was_training = model.training
    model.train()

    for bi, batch in enumerate(tqdm(loader, desc="ANP channel scoring")):
        if bi >= int(cfg.max_batches):
            break
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        with ActivationCatcher(model, layer_names, retain_grad=True) as ac:
            loss = supervised_yolo_loss(model, batch)
        loss.backward()
        for name, feat in ac.features.items():
            if feat.grad is None or feat.ndim != 4:
                continue
            act = feat.detach().abs().mean(dim=(0, 2, 3)).float().cpu()
            grad = feat.grad.detach().abs().mean(dim=(0, 2, 3)).float().cpu()
            ag = (feat.detach() * feat.grad.detach()).abs().mean(dim=(0, 2, 3)).float().cpu()
            if name not in sums:
                sums[name] = {"act": torch.zeros_like(act), "grad": torch.zeros_like(grad), "ag": torch.zeros_like(ag)}
            sums[name]["act"] += act
            sums[name]["grad"] += grad
            sums[name]["ag"] += ag
            counts[name] += 1
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    rows: List[Dict[str, Any]] = []
    for name, vals in sums.items():
        n = max(1, counts.get(name, 1))
        act = vals["act"].numpy() / n
        grad = vals["grad"].numpy() / n
        ag = vals["ag"].numpy() / n
        rarity = act / (np.mean(act) + 1e-8)
        score = ag * np.log1p(grad / (np.mean(grad) + 1e-8)) * np.log1p(rarity)
        if len(score) > cfg.max_channels_per_layer:
            keep = np.argsort(score)[-cfg.max_channels_per_layer:]
        else:
            keep = np.arange(len(score))
        for ch in keep:
            rows.append(
                {
                    "module": name,
                    "channel": int(ch),
                    "anp_score": float(score[ch]),
                    "score": float(score[ch]),
                    "mean_abs_activation": float(act[ch]),
                    "mean_abs_gradient": float(grad[ch]),
                    "mean_abs_act_x_grad": float(ag[ch]),
                    "activation_rarity": float(rarity[ch]),
                    "method": "anp_grad_sensitivity",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def merge_channel_scores(*dfs: pd.DataFrame, weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Merge ANP/FMP/correlation channel score CSVs into one ranked table."""
    dfs = [df.copy() for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    if weights is None:
        weights = [1.0] * len(dfs)
    parts: List[pd.DataFrame] = []
    for df, w in zip(dfs, weights):
        tmp = df.copy()
        if "score" not in tmp.columns:
            numeric = [c for c in tmp.columns if c.endswith("score") or c == "anp_score" or c == "fmp_score"]
            tmp["score"] = tmp[numeric[0]] if numeric else 0.0
        s = tmp["score"].astype(float).to_numpy()
        if np.nanmax(s) > np.nanmin(s):
            s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-8)
        tmp["score_norm"] = s * float(w)
        parts.append(tmp[["module", "channel", "score_norm"]])
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.groupby(["module", "channel"], as_index=False)["score_norm"].sum()
    merged = merged.rename(columns={"score_norm": "score"}).sort_values("score", ascending=False).reset_index(drop=True)
    return merged
