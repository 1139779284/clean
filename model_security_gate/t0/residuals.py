from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .metrics import extract_asr_matrix


@dataclass(frozen=True)
class ResidualRecommendation:
    attack: str
    asr: float
    phase: str
    rationale: str
    primary_losses: tuple[str, ...]
    default_weight_hint: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "asr": float(self.asr),
            "phase": self.phase,
            "rationale": self.rationale,
            "primary_losses": list(self.primary_losses),
            "default_weight_hint": float(self.default_weight_hint),
        }


def recommend_phase_for_attack(attack: str, asr: float) -> ResidualRecommendation:
    low = str(attack).lower()
    a = float(asr)
    if "wanet" in low or "warp" in low or "geometry" in low:
        return ResidualRecommendation(
            attack=attack,
            asr=a,
            phase="geometry_consistency_and_roi_stability",
            rationale="Largest residual looks geometric; prioritize paired clean/warped feature stability and target-absent non-expansion.",
            primary_losses=("paired_geometry_feature_stability", "target_absent_nonexpansion", "teacher_output_stability", "l2sp"),
            default_weight_hint=2.0,
        )
    if "semantic" in low or "cleanlabel" in low or "clean-label" in low:
        return ResidualRecommendation(
            attack=attack,
            asr=a,
            phase="semantic_causal_negative_hardening",
            rationale="Residual is semantic/background shortcut; use target-absent threshold caps and outside-region teacher preservation.",
            primary_losses=("semantic_threshold_cap", "causal_background_swap", "teacher_output_stability", "l2sp"),
            default_weight_hint=1.5,
        )
    if "oda" in low or "vanish" in low:
        return ResidualRecommendation(
            attack=attack,
            asr=a,
            phase="target_present_recall_preservation",
            rationale="Residual is disappearing-object behavior; use ODA matched candidate recall and box-preserving teacher anchors.",
            primary_losses=("matched_candidate_oda", "oda_teacher_floor", "box_stability", "clean_anchor"),
            default_weight_hint=1.5,
        )
    if "oga" in low or "blend" in low or "badnet" in low:
        return ResidualRecommendation(
            attack=attack,
            asr=a,
            phase="target_absent_false_positive_suppression",
            rationale="Residual is target generation; suppress target candidates only on target-absent rows and keep target-present anchors active.",
            primary_losses=("target_absent_nonexpansion", "negative_target_candidate_suppression", "teacher_output_stability", "l2sp"),
            default_weight_hint=1.0,
        )
    return ResidualRecommendation(
        attack=attack,
        asr=a,
        phase="general_counterfactual_replay",
        rationale="Residual family unknown; collect failure exemplars and run conservative no-worse replay.",
        primary_losses=("joint_no_worse", "teacher_output_stability", "clean_anchor"),
        default_weight_hint=1.0,
    )


def rank_residuals(report: Mapping[str, Any], *, min_asr: float = 0.0) -> list[dict[str, Any]]:
    matrix = extract_asr_matrix(report)
    recs = [recommend_phase_for_attack(k, v) for k, v in matrix.items() if float(v) >= float(min_asr)]
    recs.sort(key=lambda r: r.asr, reverse=True)
    return [r.to_dict() for r in recs]


def build_frontier_plan(reports: Sequence[Mapping[str, Any]], *, min_asr: float = 0.0, top_k: int = 5) -> dict[str, Any]:
    merged: dict[str, float] = {}
    for report in reports:
        for attack, asr in extract_asr_matrix(report).items():
            merged[attack] = max(float(asr), float(merged.get(attack, 0.0)))
    rows = [recommend_phase_for_attack(k, v).to_dict() for k, v in merged.items() if v >= float(min_asr)]
    rows.sort(key=lambda x: x["asr"], reverse=True)
    phases: list[str] = []
    for row in rows:
        phase = str(row["phase"])
        if phase not in phases:
            phases.append(phase)
    return {"top_residuals": rows[: int(top_k)], "recommended_phase_order": phases, "n_attacks": len(rows)}
