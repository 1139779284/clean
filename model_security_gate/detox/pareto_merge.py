from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import torch


_YOLO_LAYER_RE = re.compile(r"(?:^|\.)(?:model)\.(\d+)(?:\.|$)")


@dataclass
class MergeReport:
    output: str
    alpha_default: float
    alpha_by_layer: Dict[str, float]
    tensors_seen: int
    tensors_merged: int
    tensors_kept_base: int
    tensors_shape_mismatch: int
    tensors_non_float: int
    base_model: str
    source_model: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NamedLayerAlphaSpec:
    name: str
    alpha_by_layer: Dict[str, float]


def parse_alpha_grid(value: str | None) -> list[float]:
    if not value:
        return [round(x / 20.0, 4) for x in range(0, 21)]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    alphas = [float(p) for p in parts]
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alphas


def _safe_spec_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return name or fallback


def parse_layer_alpha_spec(spec: str | None) -> Dict[str, float]:
    """Parse a compact layer alpha spec.

    Format examples:

    - ``"0-9:0.25,10-21:0.50,22-999:0.80"``
    - ``"backbone=0-9:0.25;neck=10-21:0.50;head=22-999:0.80"``

    The optional group name is kept only for metadata; matching is based on
    numeric YOLO module ranges.
    """
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for raw in re.split(r"[,;]", spec):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            _, item = item.split("=", 1)
        if ":" not in item:
            raise ValueError(f"layer alpha entry must contain ':', got {raw!r}")
        key, value = item.split(":", 1)
        alpha = float(value)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"layer alpha must be in [0, 1], got {alpha}")
        _parse_layer_range(key.strip())
        out[key.strip()] = alpha
    return out


def parse_named_layer_alpha_specs(value: str | None) -> list[NamedLayerAlphaSpec]:
    """Parse one or more fixed layer-alpha specs.

    Candidate specs are separated with ``|``. Each candidate may optionally use
    ``name::spec``. Example:

    ``"head_high::0-9:0.1,10-21:0.4,22-999:0.8|backbone_high::0-9:0.8,10-999:0.2"``
    """
    if not value:
        return []
    specs: list[NamedLayerAlphaSpec] = []
    for idx, raw in enumerate(value.split("|"), start=1):
        item = raw.strip()
        if not item:
            continue
        if "::" in item:
            name, spec = item.split("::", 1)
            safe_name = _safe_spec_name(name, f"layer_{idx}")
        else:
            spec = item
            safe_name = f"layer_{idx}"
        specs.append(NamedLayerAlphaSpec(name=safe_name, alpha_by_layer=parse_layer_alpha_spec(spec)))
    return specs


def generate_group_layer_alpha_specs(
    group_alphas: Sequence[float],
    *,
    backbone_range: str = "0-9",
    neck_range: str = "10-21",
    head_range: str = "22-999",
    max_candidates: int | None = None,
) -> list[NamedLayerAlphaSpec]:
    """Generate coarse YOLO backbone/neck/head layer interpolation specs.

    The order is deterministic and prioritizes candidates where at least one
    group is close to the ASR-suppressing source model and one group is close
    to the mAP-preserving base model. This gives useful coarse layer-graft
    probes without requiring a huge full grid by default.
    """
    alphas = [float(a) for a in group_alphas]
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"group alpha must be in [0, 1], got {alpha}")

    candidates: list[tuple[float, NamedLayerAlphaSpec]] = []
    for backbone_alpha in alphas:
        for neck_alpha in alphas:
            for head_alpha in alphas:
                spread = max(backbone_alpha, neck_alpha, head_alpha) - min(backbone_alpha, neck_alpha, head_alpha)
                source_strength = backbone_alpha + neck_alpha + head_alpha
                name = (
                    f"bb{str(backbone_alpha).replace('.', 'p')}_"
                    f"neck{str(neck_alpha).replace('.', 'p')}_"
                    f"head{str(head_alpha).replace('.', 'p')}"
                )
                spec = {
                    backbone_range: float(backbone_alpha),
                    neck_range: float(neck_alpha),
                    head_range: float(head_alpha),
                }
                # High spread first, then mid-strength candidates. Pure all-0
                # or all-1 are already covered by global alpha search.
                priority = (-spread, abs(source_strength - 1.5), source_strength)
                candidates.append((priority, NamedLayerAlphaSpec(name=name, alpha_by_layer=spec)))  # type: ignore[arg-type]
    candidates.sort(key=lambda item: item[0])
    out = [spec for _, spec in candidates]
    if max_candidates is not None and max_candidates > 0:
        return out[: int(max_candidates)]
    return out


def _parse_layer_range(value: str) -> tuple[int, int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    layer = int(value)
    return layer, layer


def layer_index_for_state_key(key: str) -> int | None:
    match = _YOLO_LAYER_RE.search(key)
    if not match:
        return None
    return int(match.group(1))


def alpha_for_state_key(key: str, default_alpha: float, alpha_by_layer: Mapping[str, float] | None = None) -> float:
    if not alpha_by_layer:
        return float(default_alpha)
    layer = layer_index_for_state_key(key)
    if layer is None:
        return float(default_alpha)
    for layer_range, alpha in alpha_by_layer.items():
        lo, hi = _parse_layer_range(layer_range)
        if lo <= layer <= hi:
            return float(alpha)
    return float(default_alpha)


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _state_dict_holder(checkpoint: Any) -> tuple[Any, MutableMapping[str, torch.Tensor]]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, checkpoint.state_dict()
    if isinstance(checkpoint, MutableMapping):
        for key in ("model", "ema"):
            module = checkpoint.get(key)
            if isinstance(module, torch.nn.Module):
                return module, module.state_dict()
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint, checkpoint
    raise TypeError("Unsupported checkpoint format; expected Ultralytics checkpoint, nn.Module, or state_dict")


def _load_state_into_checkpoint(checkpoint: Any, merged: Mapping[str, torch.Tensor]) -> Any:
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.load_state_dict(merged, strict=False)
        return checkpoint
    if isinstance(checkpoint, MutableMapping):
        loaded_any = False
        for key in ("model", "ema"):
            module = checkpoint.get(key)
            if isinstance(module, torch.nn.Module):
                module.load_state_dict(merged, strict=False)
                loaded_any = True
        if not loaded_any and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            checkpoint.clear()
            checkpoint.update(merged)
        checkpoint["pareto_merge"] = checkpoint.get("pareto_merge", {})
        return checkpoint
    return checkpoint


def interpolate_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    alpha: float,
    alpha_by_layer: Mapping[str, float] | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    merged: Dict[str, torch.Tensor] = {}
    stats = {
        "tensors_seen": 0,
        "tensors_merged": 0,
        "tensors_kept_base": 0,
        "tensors_shape_mismatch": 0,
        "tensors_non_float": 0,
    }
    for key, base_tensor in base_state.items():
        stats["tensors_seen"] += 1
        source_tensor = source_state.get(key)
        if source_tensor is None or tuple(source_tensor.shape) != tuple(base_tensor.shape):
            merged[key] = base_tensor.detach().clone()
            stats["tensors_kept_base"] += 1
            if source_tensor is not None:
                stats["tensors_shape_mismatch"] += 1
            continue
        if not torch.is_floating_point(base_tensor) or not torch.is_floating_point(source_tensor):
            merged[key] = base_tensor.detach().clone()
            stats["tensors_kept_base"] += 1
            stats["tensors_non_float"] += 1
            continue
        key_alpha = alpha_for_state_key(key, default_alpha=alpha, alpha_by_layer=alpha_by_layer)
        merged[key] = (float(key_alpha) * source_tensor.detach().float() + (1.0 - float(key_alpha)) * base_tensor.detach().float()).to(
            dtype=base_tensor.dtype
        )
        stats["tensors_merged"] += 1
    return merged, stats


def interpolate_checkpoints(
    base_model: str | Path,
    source_model: str | Path,
    output_model: str | Path,
    alpha: float,
    alpha_by_layer: Mapping[str, float] | None = None,
) -> MergeReport:
    """Interpolate two compatible Ultralytics/torch checkpoints.

    ``base_model`` is the mAP-preserving side and ``source_model`` is the
    ASR-suppressing side. ``alpha=0`` equals base; ``alpha=1`` equals source.
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base_path = Path(base_model)
    source_path = Path(source_model)
    out_path = Path(output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_ckpt = _torch_load(base_path)
    source_ckpt = _torch_load(source_path)
    _, base_state = _state_dict_holder(base_ckpt)
    _, source_state = _state_dict_holder(source_ckpt)
    merged, stats = interpolate_state_dicts(base_state, source_state, alpha=alpha, alpha_by_layer=alpha_by_layer)

    out_ckpt = copy.deepcopy(base_ckpt)
    out_ckpt = _load_state_into_checkpoint(out_ckpt, merged)
    if isinstance(out_ckpt, MutableMapping):
        out_ckpt["pareto_merge"] = {
            "base_model": str(base_path),
            "source_model": str(source_path),
            "alpha_default": float(alpha),
            "alpha_by_layer": dict(alpha_by_layer or {}),
        }
    torch.save(out_ckpt, out_path)
    report = MergeReport(
        output=str(out_path),
        alpha_default=float(alpha),
        alpha_by_layer=dict(alpha_by_layer or {}),
        tensors_seen=int(stats["tensors_seen"]),
        tensors_merged=int(stats["tensors_merged"]),
        tensors_kept_base=int(stats["tensors_kept_base"]),
        tensors_shape_mismatch=int(stats["tensors_shape_mismatch"]),
        tensors_non_float=int(stats["tensors_non_float"]),
        base_model=str(base_path),
        source_model=str(source_path),
    )
    (out_path.with_suffix(out_path.suffix + ".merge.json")).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def generate_full_alpha_candidates(
    base_model: str | Path,
    source_model: str | Path,
    out_dir: str | Path,
    alphas: Sequence[float],
    prefix: str = "merge",
) -> list[MergeReport]:
    reports: list[MergeReport] = []
    for alpha in alphas:
        safe_alpha = str(float(alpha)).replace(".", "p")
        output = Path(out_dir) / f"{prefix}_alpha_{safe_alpha}.pt"
        reports.append(interpolate_checkpoints(base_model, source_model, output, alpha=float(alpha)))
    return reports


def write_merge_manifest(path: str | Path, reports: Iterable[MergeReport], extra: Mapping[str, Any] | None = None) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reports": [r.to_dict() for r in reports]}
    if extra:
        payload.update(dict(extra))
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
