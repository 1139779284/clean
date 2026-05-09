from argparse import Namespace

from scripts.run_t0_multi_attack_detox_yolo import cmd_for
from scripts.t0_evidence_pipeline import build_pipeline_commands


def test_t0_detox_plan_uses_run_external_hard_suite_roots_arg():
    args = Namespace(
        data_yaml="data.yaml",
        out="runs/out",
        external_roots=["bench"],
        target_classes=["helmet"],
        device="0",
    )
    cmd = cmd_for({"profile": "guard_free_eval", "name": "stage_00"}, args, "model.pt")
    assert "--roots" in cmd
    assert "--external-roots" not in cmd
    assert "--data-yaml" in cmd


def test_t0_evidence_pipeline_builds_all_green_views():
    args = Namespace(
        model="after.pt",
        poisoned_model="before.pt",
        data_yaml="data.yaml",
        benchmark_root="bench_full",
        trigger_only_root="bench_trigger",
        heldout_roots=["held"],
        target_classes=["helmet"],
        target_class_id=0,
        out="runs/t0",
        imgsz=416,
        conf=0.25,
        iou=0.7,
        match_iou=0.3,
        batch=16,
        workers=0,
        device="0",
        apply_overlap_class_guard=True,
        overlap_guard_suppressor_class_ids=["1"],
    )
    commands, paths = build_pipeline_commands(args)
    names = [row["name"] for row in commands]
    assert names == [
        "benchmark_audit",
        "guard_free_external",
        "guarded_external",
        "trigger_only_external",
        "clean_before",
        "clean_after",
        "evidence_pack",
    ]
    assert "--roots" in commands[1]["cmd"]
    assert "--apply-overlap-class-guard" in commands[2]["cmd"]
    assert paths["evidence_pack"].endswith("t0_evidence_pack.json")
