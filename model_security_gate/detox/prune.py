from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch


def _get_ultralytics_yolo(adapter_or_model):
    if hasattr(adapter_or_model, "model") and hasattr(adapter_or_model.model, "save"):
        return adapter_or_model.model
    if hasattr(adapter_or_model, "save"):
        return adapter_or_model
    raise TypeError("Expected UltralyticsYOLOAdapter or ultralytics.YOLO object")


def _get_torch_model(adapter_or_model):
    yolo = _get_ultralytics_yolo(adapter_or_model)
    if not hasattr(yolo, "model"):
        raise TypeError("Ultralytics YOLO object has no .model torch module")
    return yolo.model


def _module_dict(torch_model) -> Dict[str, torch.nn.Module]:
    return dict(torch_model.named_modules())


def zero_out_ranked_channels(
    adapter_or_yolo,
    ranked_channels: pd.DataFrame,
    top_k: int = 50,
    min_score: float | None = None,
) -> List[Tuple[str, int]]:
    """Soft-ablate Conv2d output channels by setting their filters to zero.

    This is conservative and architecture-compatible: it does not change tensor
    shapes. If a Conv2d belongs to an Ultralytics Conv block named '...conv', the
    sibling BN parameters are also zeroed when present.
    """
    torch_model = _get_torch_model(adapter_or_yolo)
    mods = _module_dict(torch_model)
    df = ranked_channels.copy()
    if min_score is not None and "score" in df:
        df = df[df["score"] >= min_score]
    if "score" in df:
        df = df.sort_values("score", ascending=False)
    df = df.head(top_k)
    zeroed: List[Tuple[str, int]] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            name = str(row["module"])
            ch = int(row["channel"])
            mod = mods.get(name)
            if not isinstance(mod, torch.nn.Conv2d):
                continue
            if ch < 0 or ch >= mod.out_channels:
                continue
            mod.weight[ch].zero_()
            if mod.bias is not None:
                mod.bias[ch].zero_()
            # Ultralytics Conv wrapper often has module name ending with '.conv' and sibling '.bn'.
            if name.endswith(".conv"):
                parent_name = name[: -len(".conv")]
                bn = mods.get(parent_name + ".bn")
                if isinstance(bn, torch.nn.BatchNorm2d) and ch < bn.num_features:
                    bn.weight[ch].zero_()
                    bn.bias[ch].zero_()
                    bn.running_mean[ch].zero_()
                    bn.running_var[ch].fill_(1.0)
            zeroed.append((name, ch))
    return zeroed


def save_ultralytics_model(adapter_or_yolo, output_path: str | Path) -> Path:
    yolo = _get_ultralytics_yolo(adapter_or_yolo)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(output_path))
    return output_path
