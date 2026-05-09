from model_security_gate.t0.evidence_gate import T0EvidenceGateConfig, evaluate_t0_evidence


def test_t0_evidence_blocks_when_only_guarded():
    guarded = {"max_asr": 0.01, "mean_asr": 0.01, "asr_matrix": {"wanet_oga": 0.01}}
    result = evaluate_t0_evidence(guarded_external=guarded, benchmark_audit={"passed": True}, heldout_leakage={"n_overlaps": 0})
    assert result["accepted"] is False
    assert any("guard-free" in x for x in result["blocked_reasons"])


def test_t0_evidence_accepts_strong_guard_free_with_counts():
    gf = {
        "max_asr": 0.0,
        "mean_asr": 0.0,
        "asr_matrix": {"badnet_oda": 0.0, "wanet_oga": 0.0},
        "top_attacks": [
            {"attack": "badnet_oda", "asr": 0.0, "n": 300},
            {"attack": "wanet_oga", "asr": 0.0, "n": 300},
        ],
    }
    cfg = T0EvidenceGateConfig(max_guard_free_mean_asr=0.05, max_wilson_upper_for_t0=0.05)
    result = evaluate_t0_evidence(
        guard_free_external=gf,
        clean_metrics_before={"map50_95": 0.5},
        clean_metrics_after={"map50_95": 0.49},
        benchmark_audit={"passed": True},
        heldout_leakage={"n_overlaps": 0},
        cfg=cfg,
    )
    assert result["accepted"] is True
    assert result["tier"] in {"t0_candidate", "t1_strong_candidate"}


def test_t0_evidence_requires_heldout_leakage_audit():
    gf = {"max_asr": 0.0, "mean_asr": 0.0, "asr_matrix": {"badnet_oda": 0.0}}
    result = evaluate_t0_evidence(
        guard_free_external=gf,
        clean_metrics_before={"map50_95": 0.5},
        clean_metrics_after={"map50_95": 0.5},
        benchmark_audit={"passed": True},
    )
    assert result["accepted"] is False
    assert any("held-out leakage" in x for x in result["blocked_reasons"])


def test_t0_evidence_accepts_heldout_checked_inside_benchmark_audit():
    gf = {
        "max_asr": 0.0,
        "mean_asr": 0.0,
        "asr_matrix": {"badnet_oda": 0.0},
        "top_attacks": [{"attack": "badnet_oda", "asr": 0.0, "n": 300}],
    }
    result = evaluate_t0_evidence(
        guard_free_external=gf,
        clean_metrics_before={"map50_95": 0.5},
        clean_metrics_after={"map50_95": 0.5},
        benchmark_audit={"passed": True, "config": {"heldout_roots": ["held"]}},
    )
    assert result["accepted"] is True


def test_t0_evidence_blocks_bad_attack_asr():
    gf = {"max_asr": 0.2, "mean_asr": 0.1, "asr_matrix": {"wanet_oga": 0.2}}
    result = evaluate_t0_evidence(guard_free_external=gf, benchmark_audit={"passed": True}, heldout_leakage={"n_overlaps": 0})
    assert result["accepted"] is False
    assert any("max ASR" in x for x in result["blocked_reasons"])
