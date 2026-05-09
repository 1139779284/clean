from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Any


_Z_FOR_CONFIDENCE = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.975: 2.241402727604947,
    0.99: 2.5758293035489004,
}


@dataclass(frozen=True)
class WilsonInterval:
    successes: int
    total: int
    rate: float
    low: float
    high: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, float | int]:
        return {
            "successes": int(self.successes),
            "total": int(self.total),
            "rate": float(self.rate),
            "low": float(self.low),
            "high": float(self.high),
            "confidence": float(self.confidence),
        }


def z_value(confidence: float = 0.95) -> float:
    """Return a normal z value for common two-sided Wilson intervals.

    The project avoids a SciPy dependency for CI smoke.  Common confidence
    levels are exact table values; uncommon levels use a conservative fallback
    close to 95%.
    """

    c = float(confidence)
    if c in _Z_FOR_CONFIDENCE:
        return _Z_FOR_CONFIDENCE[c]
    # Conservative fallback: do not overstate precision for unusual inputs.
    if c >= 0.99:
        return _Z_FOR_CONFIDENCE[0.99]
    if c >= 0.975:
        return _Z_FOR_CONFIDENCE[0.975]
    if c >= 0.90:
        return _Z_FOR_CONFIDENCE[0.95]
    return _Z_FOR_CONFIDENCE[0.90]


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> WilsonInterval:
    s = int(successes)
    n = int(total)
    if n <= 0:
        return WilsonInterval(s, n, 0.0, 0.0, 1.0, float(confidence))
    s = max(0, min(s, n))
    p = s / n
    z = z_value(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return WilsonInterval(s, n, p, max(0.0, center - half), min(1.0, center + half), float(confidence))


def zero_failure_upper_bound(total: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound for zero observed failures.

    If zero failures are observed in n Bernoulli trials, the upper confidence
    bound p satisfies (1 - p)^n = alpha.  This is useful for avoiding the common
    mistake of reporting zero ASR as proof of zero risk on small suites.
    """

    n = int(total)
    if n <= 0:
        return 1.0
    alpha = max(1e-12, min(1.0, 1.0 - float(confidence)))
    return 1.0 - alpha ** (1.0 / n)


def required_zero_failure_n(max_rate: float, confidence: float = 0.95) -> int:
    """Samples required so zero failures upper bound is <= max_rate."""

    p = float(max_rate)
    if p <= 0:
        raise ValueError("max_rate must be positive")
    if p >= 1:
        return 1
    alpha = max(1e-12, min(1.0, 1.0 - float(confidence)))
    return int(math.ceil(math.log(alpha) / math.log(1.0 - p)))


def summarize_binomial_rows(rows: Iterable[Mapping[str, Any]], *, success_key: str = "success", confidence: float = 0.95) -> dict[str, Any]:
    """Summarize boolean row outcomes with Wilson confidence bounds."""

    total = 0
    successes = 0
    for row in rows:
        total += 1
        value = row.get(success_key, False)
        if isinstance(value, str):
            ok = value.strip().lower() in {"1", "true", "yes", "y", "success", "fail", "failure"}
        else:
            ok = bool(value)
        successes += int(ok)
    interval = wilson_interval(successes, total, confidence)
    return interval.to_dict()
