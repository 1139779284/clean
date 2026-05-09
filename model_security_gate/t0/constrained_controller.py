from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class ConstraintSpec:
    name: str
    limit: float
    direction: str = "max"  # max or min
    lambda_value: float = 1.0
    growth: float = 1.5
    shrink: float = 0.90
    min_lambda: float = 0.0
    max_lambda: float = 1000.0

    def violation(self, value: float) -> float:
        v = float(value)
        if self.direction == "min":
            return max(0.0, float(self.limit) - v)
        return max(0.0, v - float(self.limit))


@dataclass
class ConstraintUpdate:
    name: str
    value: float | None
    limit: float
    violation: float
    old_lambda: float
    new_lambda: float
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LagrangianNoWorseController:
    """Small adaptive controller for first-order no-worse training loops.

    The trainer can update this object after each cheap proxy-eval window.  It
    is intentionally framework-agnostic so it can drive PyTorch losses, YOLO
    callback weights, or staged repair profile selection.
    """

    constraints: dict[str, ConstraintSpec] = field(default_factory=dict)
    tolerance: float = 0.0

    def update(self, metrics: Mapping[str, Any]) -> list[ConstraintUpdate]:
        updates: list[ConstraintUpdate] = []
        for name, spec in self.constraints.items():
            raw = metrics.get(name)
            old = float(spec.lambda_value)
            if raw is None:
                new = min(spec.max_lambda, max(spec.min_lambda, old * spec.growth))
                spec.lambda_value = new
                updates.append(ConstraintUpdate(name, None, spec.limit, float("inf"), old, new, True))
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            violation = spec.violation(value)
            active = violation > float(self.tolerance)
            if active:
                new = min(spec.max_lambda, max(spec.min_lambda, old * spec.growth + violation))
            else:
                new = min(spec.max_lambda, max(spec.min_lambda, old * spec.shrink))
            spec.lambda_value = new
            updates.append(ConstraintUpdate(name, value, spec.limit, violation, old, new, active))
        return updates

    def weights(self) -> dict[str, float]:
        return {name: float(spec.lambda_value) for name, spec in self.constraints.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"constraints": {k: asdict(v) for k, v in self.constraints.items()}, "tolerance": float(self.tolerance)}


def default_t0_controller() -> LagrangianNoWorseController:
    return LagrangianNoWorseController(
        constraints={
            "guard_free_max_asr": ConstraintSpec("guard_free_max_asr", 0.05, "max", lambda_value=5.0),
            "wanet_oga": ConstraintSpec("wanet_oga", 0.05, "max", lambda_value=3.0),
            "semantic_green_cleanlabel": ConstraintSpec("semantic_green_cleanlabel", 0.05, "max", lambda_value=3.0),
            "badnet_oda": ConstraintSpec("badnet_oda", 0.05, "max", lambda_value=2.0),
            "map50_95_drop": ConstraintSpec("map50_95_drop", 0.03, "max", lambda_value=4.0),
        },
        tolerance=1e-6,
    )
