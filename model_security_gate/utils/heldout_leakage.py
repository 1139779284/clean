from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class HeldoutLeak:
    candidate: str
    heldout_root: str
    reason: str


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_within_path(candidate: str | Path, root: str | Path) -> bool:
    candidate_path = _resolve_path(candidate)
    root_path = _resolve_path(root)
    try:
        candidate_path.relative_to(root_path)
        return True
    except ValueError:
        return False


def find_path_leaks(
    candidates: Iterable[str | Path],
    heldout_roots: Iterable[str | Path],
) -> list[HeldoutLeak]:
    leaks: list[HeldoutLeak] = []
    roots = [str(_resolve_path(root)) for root in heldout_roots]
    for candidate in candidates:
        candidate_path = str(_resolve_path(candidate))
        for root in roots:
            if is_within_path(candidate_path, root):
                leaks.append(
                    HeldoutLeak(
                        candidate=candidate_path,
                        heldout_root=root,
                        reason="candidate_path_inside_heldout_root",
                    )
                )
    return leaks


def scan_text_for_heldout_paths(
    text: str,
    heldout_roots: Iterable[str | Path],
    source: str,
) -> list[HeldoutLeak]:
    normalized_text = text.replace("\\", "/").casefold()
    leaks: list[HeldoutLeak] = []
    for root in heldout_roots:
        root_path = str(_resolve_path(root))
        normalized_root = root_path.replace("\\", "/").casefold()
        if normalized_root in normalized_text:
            leaks.append(
                HeldoutLeak(
                    candidate=source,
                    heldout_root=root_path,
                    reason="heldout_root_string_found_in_manifest",
                )
            )
    return leaks

