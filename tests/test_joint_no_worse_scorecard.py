from model_security_gate.detox.joint_no_worse import JointNoWorseConfig, candidate_no_worse_scorecard


def test_joint_no_worse_blocks_semantic_residual():
    metrics = {
        "badnet_oda": 0.05,
        "semantic_green_cleanlabel": 0.05,
        "blend_oga": 0.0,
        "wanet_oga": 0.0,
        "semantic_fp_max_conf": 0.4411,
        "map50_95": 0.1703,
    }
    scorecard = candidate_no_worse_scorecard(metrics, JointNoWorseConfig.production_defaults().specs)
    assert not scorecard.accepted
    assert any("semantic_green_cleanlabel" in b for b in scorecard.blockers)
    assert any("semantic_fp_max_conf" in b for b in scorecard.blockers)
