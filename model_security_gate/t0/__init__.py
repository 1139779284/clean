"""T0 research-grade evidence, benchmark governance, and frontier planning tools.

These modules are intentionally lightweight and detector-agnostic.  They do not
replace the existing Model Security Gate training code; they add a stricter
research protocol around it so that guarded engineering results, guard-free
weight purification, trigger-only ASR, leakage checks, and statistical evidence
are reported separately.
"""

from .stats import WilsonInterval, wilson_interval, zero_failure_upper_bound, required_zero_failure_n
from .evidence_gate import T0EvidenceGateConfig, evaluate_t0_evidence
from .matrix_aggregator import (
    MatrixAggregatorConfig,
    aggregate_matrix_entries,
    aggregate_matrix_summary,
    render_matrix_aggregate_markdown,
    write_matrix_aggregate,
)
from .defense_leaderboard import (
    DefenseEntry,
    DefenseLeaderboardConfig,
    build_defense_leaderboard,
    evaluate_defense_entry,
    load_entries_from_manifest,
    mcnemar_exact_pvalue,
    render_defense_leaderboard_markdown,
    write_defense_leaderboard,
)
from .defense_certificate import (
    DefenseCertificateConfig,
    build_defense_certificates,
    certify_defense_entry,
    holm_bonferroni_adjust,
    render_defense_certificates_latex,
    render_defense_certificates_markdown,
    write_defense_certificates,
)
from .leakage_audit import (
    LeakageAudit,
    audit_cfrc_manifest,
    audit_hybrid_manifest_against_eval,
)
from .ablation_plan import (
    AblationArm,
    DetoxAblationSpec,
    PlannedRun,
    build_arm_train_command,
    build_cfrc_manifest,
    default_contribution_1_arms,
    hybrid_manifest_to_defense_entry,
    plan_runs,
    render_runbook_markdown,
    write_ablation_plan,
)

__all__ = [
    "WilsonInterval",
    "wilson_interval",
    "zero_failure_upper_bound",
    "required_zero_failure_n",
    "T0EvidenceGateConfig",
    "evaluate_t0_evidence",
    "MatrixAggregatorConfig",
    "aggregate_matrix_entries",
    "aggregate_matrix_summary",
    "render_matrix_aggregate_markdown",
    "write_matrix_aggregate",
    "DefenseEntry",
    "DefenseLeaderboardConfig",
    "build_defense_leaderboard",
    "evaluate_defense_entry",
    "load_entries_from_manifest",
    "mcnemar_exact_pvalue",
    "render_defense_leaderboard_markdown",
    "write_defense_leaderboard",
    "DefenseCertificateConfig",
    "build_defense_certificates",
    "certify_defense_entry",
    "holm_bonferroni_adjust",
    "render_defense_certificates_markdown",
    "render_defense_certificates_latex",
    "write_defense_certificates",
    "LeakageAudit",
    "audit_cfrc_manifest",
    "audit_hybrid_manifest_against_eval",
    "AblationArm",
    "DetoxAblationSpec",
    "PlannedRun",
    "build_arm_train_command",
    "build_cfrc_manifest",
    "default_contribution_1_arms",
    "hybrid_manifest_to_defense_entry",
    "plan_runs",
    "render_runbook_markdown",
    "write_ablation_plan",
]
