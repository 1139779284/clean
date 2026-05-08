from model_security_gate.guard.semantic_abstain import SemanticAbstainRule, decide_semantic_abstain
from model_security_gate.adapters.base import Detection
from model_security_gate.detox.external_hard_suite import apply_semantic_abstain_guard_to_detections


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


def test_external_guard_removes_matched_detection_only():
    rules = [
        SemanticAbstainRule(
            rule_id="helmet_fp",
            class_name="helmet",
            min_conf=0.25,
            image_globs=["*attack_0011*"],
        )
    ]
    detections = [
        Detection(xyxy=(1, 1, 10, 10), conf=0.44, cls_id=0, cls_name="helmet"),
        Detection(xyxy=(20, 20, 40, 40), conf=0.91, cls_id=1, cls_name="head"),
    ]

    kept, info = apply_semantic_abstain_guard_to_detections(detections, rules, image_path="attack_0011_helm_021400.jpg")

    assert info["action"] == "review"
    assert info["removed_detections"] == 1
    assert [d.cls_name for d in kept] == ["head"]
