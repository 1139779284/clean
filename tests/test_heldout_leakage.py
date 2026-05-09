from pathlib import Path

from model_security_gate.utils.heldout_leakage import (
    find_path_leaks,
    scan_text_for_heldout_paths,
)


def test_find_path_leaks_detects_child_path(tmp_path: Path) -> None:
    heldout = tmp_path / "try_attack_data"
    child = heldout / "image.jpg"
    heldout.mkdir()
    child.write_text("x", encoding="utf-8")

    leaks = find_path_leaks([child], [heldout])

    assert len(leaks) == 1
    assert leaks[0].reason == "candidate_path_inside_heldout_root"


def test_scan_text_for_heldout_paths_detects_manifest_reference(tmp_path: Path) -> None:
    heldout = tmp_path / "try_attack_data"
    heldout.mkdir()
    text = f'{{"images": "{heldout}"}}'

    leaks = scan_text_for_heldout_paths(text, [heldout], "manifest.json")

    assert len(leaks) == 1
    assert leaks[0].reason == "heldout_root_string_found_in_manifest"

