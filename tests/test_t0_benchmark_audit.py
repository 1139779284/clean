from pathlib import Path

from model_security_gate.t0.benchmark_audit import BenchmarkAuditConfig, audit_benchmark


def _write_case(root: Path, attack: str, cls: int | None) -> None:
    img = root / "data" / attack / "images" / "val" / f"{attack}_001.jpg"
    lab = root / "data" / attack / "labels" / "val" / f"{attack}_001.txt"
    img.parent.mkdir(parents=True, exist_ok=True)
    lab.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(f"fake-{attack}".encode())
    if cls is None:
        lab.write_text("", encoding="utf-8")
    else:
        lab.write_text(f"{cls} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_benchmark_audit_goal_policy_passes(tmp_path):
    root = tmp_path / "bench"
    _write_case(root, "badnet_oda", 0)
    _write_case(root, "wanet_oga", 1)
    result = audit_benchmark(BenchmarkAuditConfig(roots=[str(root)], target_class_id=0))
    assert result["passed"] is True
    assert result["n_attacks"] == 2


def test_benchmark_audit_blocks_no_evaluable_oga_negatives(tmp_path):
    root = tmp_path / "bench"
    _write_case(root, "wanet_oga", 0)
    result = audit_benchmark(BenchmarkAuditConfig(roots=[str(root)], target_class_id=0))
    assert result["passed"] is False
    assert any(f["code"] == "no_evaluable_target_absent_rows" for f in result["findings"])


def test_benchmark_audit_blocks_heldout_overlap(tmp_path):
    root = tmp_path / "bench"
    held = tmp_path / "held"
    _write_case(root, "badnet_oda", 0)
    held.mkdir()
    (held / "copy.jpg").write_bytes(b"fake-badnet_oda")
    result = audit_benchmark(BenchmarkAuditConfig(roots=[str(root)], target_class_id=0, heldout_roots=[str(held)]))
    assert result["passed"] is False
    assert any(f["code"] == "heldout_overlap" for f in result["findings"])
