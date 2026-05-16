"""LaTeX rendering for CFRC certificates."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_security_gate.t0.defense_certificate import (
    DefenseCertificateConfig,
    build_defense_certificates,
    render_defense_certificates_latex,
    write_defense_certificates,
)
from model_security_gate.t0.defense_leaderboard import DefenseEntry


def _report(attack_to_mask: dict[str, list[bool]]) -> dict:
    rows = []
    asr_matrix = {}
    for attack, flags in attack_to_mask.items():
        asr_matrix[attack] = sum(1 for f in flags if f) / max(1, len(flags))
        for i, success in enumerate(flags):
            rows.append(
                {
                    "attack": attack,
                    "image_basename": f"{attack}_{i:03d}.jpg",
                    "success": bool(success),
                }
            )
    return {"summary": {"asr_matrix": asr_matrix}, "rows": rows}


def _payload() -> dict:
    entry = DefenseEntry(
        name="demo",
        poisoned_model_id="p",
        defense="demo",
        poisoned_external=_report({"badnet_oga": [True] * 30 + [False] * 70}),
        defended_external=_report({"badnet_oga": [False] * 100}),
        clean_before={"map50_95": 0.30},
        clean_after={"map50_95": 0.30},
    )
    return build_defense_certificates([entry], cfg=DefenseCertificateConfig(n_bootstrap=200))


def test_latex_output_contains_booktabs_and_caption() -> None:
    payload = _payload()
    tex = render_defense_certificates_latex(payload)
    assert "\\begin{table}" in tex
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    assert "\\caption{" in tex
    assert "CFRC" in tex


def test_latex_contains_per_attack_detail_for_every_entry() -> None:
    payload = _payload()
    tex = render_defense_certificates_latex(payload)
    # One row in ranking + per-attack table for each entry.
    assert tex.count("\\begin{table}") == 2
    assert "badnet\\_oga" in tex  # underscore escaped


def test_write_defense_certificates_emits_latex_by_default(tmp_path: Path) -> None:
    payload = _payload()
    json_path, md_path = write_defense_certificates(tmp_path, payload)
    assert json_path.exists()
    assert md_path.exists()
    assert (tmp_path / "T0_DEFENSE_CERTIFICATE.tex").exists()


def test_write_defense_certificates_can_suppress_latex(tmp_path: Path) -> None:
    payload = _payload()
    write_defense_certificates(tmp_path, payload, emit_latex=False)
    assert not (tmp_path / "T0_DEFENSE_CERTIFICATE.tex").exists()
