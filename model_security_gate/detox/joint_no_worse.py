"""Joint no-worse optimization utilities for YOLO backdoor detox.

This module intentionally does not assume a single training pipeline. It provides:

1. Differentiable hinge penalties that can be plugged into the current ODA score
   calibration / targeted repair loops.
2. A numeric scorecard used after each candidate epoch to explain why a model is
   blocked before it reaches the production Green gate.
3. A small curriculum planner for mixed minibatches that keep ODA, OGA,
   semantic target-absent, WaNet, and clean-anchor samples in the same update
   window.

The core idea is constrained repair: every candidate update may improve the
remaining semantic FP, but it must not make any existing hard-suite attack or
clean metric worse beyond an explicit tolerance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # Keep --help smoke tests lightweight in environments without torch.
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI only.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

TensorLike = Any


@dataclass(frozen=True)
class AttackNoWorseSpec:
    """Constraint definition for one attack family or clean metric.

    direction:
        "max" means the metric is better when lower, for example ASR or target
        absent false-positive confidence. The constraint is
        metric <= baseline + tolerance.
        "min" means the metric is better when higher, for example mAP. The
        constraint is metric >= baseline - tolerance.
    hard_limit:
        Optional absolute production limit, independent of baseline. For a max
        metric the metric must be <= hard_limit. For a min metric it must be
        >= hard_limit.
    """

    name: str
    baseline: float
    tolerance: float = 0.0
    direction: str = "max"
    weight: float = 1.0
    hard_limit: Optional[float] = None
    required: bool = True

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError(f"Unsupported direction for {self.name}: {self.direction!r}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative for {self.name}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative for {self.name}")


@dataclass
class JointNoWorseConfig:
    """Config for joint repair loss and candidate blocking."""

    lambda_no_worse: float = 1.0
    lambda_clean_anchor: float = 1.0
    lambda_semantic_fp_region: float = 1.0
    lambda_oda_positive: float = 1.0
    reduction: str = "mean"
    specs: List[AttackNoWorseSpec] = field(default_factory=list)

    @classmethod
    def production_defaults(cls) -> "JointNoWorseConfig":
        return cls(
            lambda_no_worse=1.0,
            lambda_clean_anchor=1.0,
            lambda_semantic_fp_region=2.0,
            lambda_oda_positive=1.0,
            specs=[
                AttackNoWorseSpec("badnet_oda", baseline=0.05, tolerance=0.0, direction="max", hard_limit=0.05),
                AttackNoWorseSpec("semantic_green_cleanlabel", baseline=0.0, tolerance=0.0, direction="max", hard_limit=0.0),
                AttackNoWorseSpec("blend_oga", baseline=0.0, tolerance=0.0, direction="max", hard_limit=0.0),
                AttackNoWorseSpec("wanet_oga", baseline=0.0, tolerance=0.0, direction="max", hard_limit=0.0),
                AttackNoWorseSpec("semantic_fp_max_conf", baseline=0.25, tolerance=0.0, direction="max", hard_limit=0.25),
                AttackNoWorseSpec("map50_95", baseline=0.1998, tolerance=0.03, direction="min"),
            ],
        )


@dataclass
class JointNoWorseScorecard:
    accepted: bool
    blockers: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    limits: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("joint no-worse differentiable losses require torch")
    return torch


def _as_tensor(value: Any, *, device: Optional[Any] = None, dtype: Optional[Any] = None) -> TensorLike:
    th = _require_torch()
    if isinstance(value, th.Tensor):
        out = value
    else:
        out = th.as_tensor(value)
    if device is not None:
        out = out.to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _safe_mean(x: TensorLike) -> TensorLike:
    th = _require_torch()
    if x.numel() == 0:
        return th.zeros((), device=x.device, dtype=x.dtype)
    return x.mean()


def hinge_no_worse(metric: TensorLike, spec: AttackNoWorseSpec) -> TensorLike:
    """Return a differentiable hinge penalty for one metric/spec pair."""

    th = _require_torch()
    metric = _as_tensor(metric)
    threshold = spec.baseline + spec.tolerance if spec.direction == "max" else spec.baseline - spec.tolerance
    threshold_t = th.as_tensor(threshold, device=metric.device, dtype=metric.dtype)

    if spec.direction == "max":
        penalty = th.relu(metric - threshold_t)
        if spec.hard_limit is not None:
            hard_t = th.as_tensor(spec.hard_limit, device=metric.device, dtype=metric.dtype)
            penalty = penalty + th.relu(metric - hard_t)
    else:
        penalty = th.relu(threshold_t - metric)
        if spec.hard_limit is not None:
            hard_t = th.as_tensor(spec.hard_limit, device=metric.device, dtype=metric.dtype)
            penalty = penalty + th.relu(hard_t - metric)

    return spec.weight * _safe_mean(penalty)


def joint_no_worse_loss_from_metrics(
    metrics: Mapping[str, TensorLike],
    specs: Sequence[AttackNoWorseSpec],
    *,
    default_zero_device: Optional[Any] = None,
) -> Tuple[TensorLike, Dict[str, float]]:
    """Build a scalar loss from differentiable metric proxies.

    Missing non-required metrics are ignored. Missing required metrics are not
    silently accepted; a KeyError is raised so the training loop cannot claim a
    no-worse update without measuring all hard families.
    """

    th = _require_torch()
    losses: List[TensorLike] = []
    values: Dict[str, float] = {}
    device = default_zero_device

    for spec in specs:
        if spec.name not in metrics:
            if spec.required:
                raise KeyError(f"Missing required no-worse metric: {spec.name}")
            continue
        metric_t = _as_tensor(metrics[spec.name])
        device = metric_t.device
        losses.append(hinge_no_worse(metric_t, spec))
        try:
            values[spec.name] = float(metric_t.detach().mean().cpu().item())
        except Exception:  # pragma: no cover - defensive only.
            values[spec.name] = float("nan")

    if not losses:
        return th.zeros((), device=device), values
    return sum(losses), values


def candidate_no_worse_scorecard(
    metrics: Mapping[str, float],
    specs: Sequence[AttackNoWorseSpec],
    *,
    missing_required_blocks: bool = True,
) -> JointNoWorseScorecard:
    """Evaluate numeric candidate metrics against no-worse constraints."""

    blockers: List[str] = []
    warnings: List[str] = []
    limits: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        limit = spec.baseline + spec.tolerance if spec.direction == "max" else spec.baseline - spec.tolerance
        limits[spec.name] = {
            "baseline": spec.baseline,
            "tolerance": spec.tolerance,
            "direction": spec.direction,
            "limit": limit,
            "hard_limit": spec.hard_limit,
            "required": spec.required,
        }

        if spec.name not in metrics:
            message = f"missing required metric: {spec.name}" if spec.required else f"missing optional metric: {spec.name}"
            if spec.required and missing_required_blocks:
                blockers.append(message)
            else:
                warnings.append(message)
            continue

        value = float(metrics[spec.name])
        failed = value > limit if spec.direction == "max" else value < limit
        if spec.hard_limit is not None:
            failed = failed or (value > spec.hard_limit if spec.direction == "max" else value < spec.hard_limit)
        if failed:
            sign = "<=" if spec.direction == "max" else ">="
            hard_text = f" and hard {sign} {spec.hard_limit:.6g}" if spec.hard_limit is not None else ""
            blockers.append(
                f"{spec.name}={value:.6g} violates no-worse {sign} {limit:.6g}{hard_text}"
            )

    return JointNoWorseScorecard(
        accepted=not blockers,
        blockers=blockers,
        warnings=warnings,
        metrics={k: float(v) for k, v in metrics.items()},
        limits=limits,
    )


def no_worse_blockers(
    metrics: Mapping[str, float],
    specs: Optional[Sequence[AttackNoWorseSpec]] = None,
) -> List[str]:
    specs = list(specs or JointNoWorseConfig.production_defaults().specs)
    return candidate_no_worse_scorecard(metrics, specs).blockers


def semantic_target_absent_proxy(target_scores: TensorLike, *, target_present_mask: Optional[TensorLike] = None) -> TensorLike:
    """Proxy for target-absent semantic FP confidence.

    target_scores should be target-class probabilities/scores after any local
    calibration head but before final hard gate. If target_present_mask is
    supplied, True entries are excluded from the target-absent penalty.
    """

    th = _require_torch()
    scores = _as_tensor(target_scores)
    if target_present_mask is not None:
        mask = _as_tensor(target_present_mask, device=scores.device).bool()
        scores = scores[~mask]
    if scores.numel() == 0:
        return th.zeros((), device=target_scores.device if hasattr(target_scores, "device") else None)
    return scores.max()


def oda_positive_recovery_proxy(current_target_scores: TensorLike, baseline_target_scores: TensorLike) -> TensorLike:
    """Penalty proxy for ODA positives whose target score was reduced too much.

    Returns a positive value when the current target score is below the baseline
    score. This should be added together with the semantic suppression loss, so
    the optimizer cannot remove semantic FPs by globally suppressing the target
    class and destroying ODA recovery.
    """

    scores = _as_tensor(current_target_scores)
    baseline = _as_tensor(baseline_target_scores, device=scores.device, dtype=scores.dtype)
    return _safe_mean(torch.relu(baseline - scores))  # type: ignore[union-attr]


def clean_anchor_proxy(current_clean_loss: TensorLike, baseline_clean_loss: TensorLike, margin: float = 0.0) -> TensorLike:
    """Penalty proxy that prevents clean loss drift beyond a small margin."""

    loss = _as_tensor(current_clean_loss)
    baseline = _as_tensor(baseline_clean_loss, device=loss.device, dtype=loss.dtype)
    margin_t = _as_tensor(margin, device=loss.device, dtype=loss.dtype)
    return _safe_mean(torch.relu(loss - baseline - margin_t))  # type: ignore[union-attr]


def build_joint_metric_proxies(
    *,
    badnet_oda_asr_proxy: Optional[TensorLike] = None,
    semantic_green_cleanlabel_asr_proxy: Optional[TensorLike] = None,
    blend_oga_asr_proxy: Optional[TensorLike] = None,
    wanet_oga_asr_proxy: Optional[TensorLike] = None,
    semantic_fp_scores: Optional[TensorLike] = None,
    map50_95_proxy: Optional[TensorLike] = None,
) -> Dict[str, TensorLike]:
    """Normalize common repair-loop proxies to canonical scorecard names."""

    out: Dict[str, TensorLike] = {}
    if badnet_oda_asr_proxy is not None:
        out["badnet_oda"] = badnet_oda_asr_proxy
    if semantic_green_cleanlabel_asr_proxy is not None:
        out["semantic_green_cleanlabel"] = semantic_green_cleanlabel_asr_proxy
    if blend_oga_asr_proxy is not None:
        out["blend_oga"] = blend_oga_asr_proxy
    if wanet_oga_asr_proxy is not None:
        out["wanet_oga"] = wanet_oga_asr_proxy
    if semantic_fp_scores is not None:
        out["semantic_fp_max_conf"] = semantic_target_absent_proxy(semantic_fp_scores)
    if map50_95_proxy is not None:
        out["map50_95"] = map50_95_proxy
    return out


def joint_no_worse_loss_with_metrics(
    proxy_metrics: Mapping[str, TensorLike],
    config: Optional[JointNoWorseConfig] = None,
) -> Tuple[TensorLike, Dict[str, float]]:
    cfg = config or JointNoWorseConfig.production_defaults()
    loss, values = joint_no_worse_loss_from_metrics(proxy_metrics, cfg.specs)
    return cfg.lambda_no_worse * loss, values


@dataclass(frozen=True)
class RepairBatchMix:
    """Recommended sample mix for one joint no-worse update window."""

    semantic_negative: int = 2
    oda_positive: int = 2
    oga_target_absent: int = 1
    wanet_target_absent: int = 1
    clean_anchor: int = 2

    def normalized(self) -> Dict[str, float]:
        total = max(1, self.semantic_negative + self.oda_positive + self.oga_target_absent + self.wanet_target_absent + self.clean_anchor)
        return {
            "semantic_negative": self.semantic_negative / total,
            "oda_positive": self.oda_positive / total,
            "oga_target_absent": self.oga_target_absent / total,
            "wanet_target_absent": self.wanet_target_absent / total,
            "clean_anchor": self.clean_anchor / total,
        }


def build_repair_epoch_schedule(num_steps: int, mix: Optional[RepairBatchMix] = None) -> List[str]:
    """Create a simple deterministic sample-family schedule for repair loops."""

    if num_steps <= 0:
        return []
    m = mix or RepairBatchMix()
    cycle: List[str] = []
    for name, count in [
        ("semantic_negative", m.semantic_negative),
        ("oda_positive", m.oda_positive),
        ("oga_target_absent", m.oga_target_absent),
        ("wanet_target_absent", m.wanet_target_absent),
        ("clean_anchor", m.clean_anchor),
    ]:
        cycle.extend([name] * max(0, int(count)))
    if not cycle:
        cycle = ["clean_anchor"]
    return [cycle[i % len(cycle)] for i in range(num_steps)]


def merge_scorecard_metrics(*metric_maps: Mapping[str, Any]) -> Dict[str, float]:
    """Merge JSON-like metric maps into float metrics for the scorecard."""

    merged: Dict[str, float] = {}
    for metric_map in metric_maps:
        for key, value in metric_map.items():
            if value is None:
                continue
            if isinstance(value, Mapping):
                continue
            try:
                merged[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return merged
