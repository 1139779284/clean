"""T0 research-grade evidence, benchmark governance, and frontier planning tools.

These modules are intentionally lightweight and detector-agnostic.  They do not
replace the existing Model Security Gate training code; they add a stricter
research protocol around it so that guarded engineering results, guard-free
weight purification, trigger-only ASR, leakage checks, and statistical evidence
are reported separately.
"""

from .stats import WilsonInterval, wilson_interval, zero_failure_upper_bound, required_zero_failure_n
from .evidence_gate import T0EvidenceGateConfig, evaluate_t0_evidence

__all__ = [
    "WilsonInterval",
    "wilson_interval",
    "zero_failure_upper_bound",
    "required_zero_failure_n",
    "T0EvidenceGateConfig",
    "evaluate_t0_evidence",
]
