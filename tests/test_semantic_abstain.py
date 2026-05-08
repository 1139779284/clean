from model_security_gate.guard.semantic_abstain import SemanticAbstainRule, decide_semantic_abstain


def test_semantic_abstain_matches_known_pattern():
    rules = [
        SemanticAbstainRule(
            rule_id="helmet_fp",
            class_name="helmet",
            min_conf=0.25,
            region_xyxy=[10, 10, 100, 100],
            require_center_in_region=True,
        )
    ]
    detections = [{"class_name": "helmet", "conf": 0.44, "xyxy": [20, 20, 40, 40]}]
    decision = decide_semantic_abstain(detections, rules, image_path="attack_0011_helm_021400.jpg")
    assert decision.action == "review"
    assert decision.matched_rules[0]["rule_id"] == "helmet_fp"


def test_semantic_abstain_passes_unmatched_detection():
    rules = [SemanticAbstainRule(rule_id="helmet_fp", class_name="helmet", min_conf=0.25)]
    detections = [{"class_name": "vest", "conf": 0.90, "xyxy": [20, 20, 40, 40]}]
    decision = decide_semantic_abstain(detections, rules)
    assert decision.action == "pass"
