from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from model_security_gate.utils.io import IMAGE_EXTS, label_path_for_image, read_yaml


@dataclass
class ImageLabelRecord:
    image_path: str
    label_path: str
    attack: str
    sha256: str
    classes: list[int] = field(default_factory=list)


@dataclass
class BenchmarkAuditConfig:
    roots: Sequence[str]
    target_class_id: int = 0
    suppressor_class_id: int | None = 1
    min_images_per_attack: int = 0
    heldout_roots: Sequence[str] = field(default_factory=tuple)
    expected_goals: Mapping[str, str] = field(default_factory=dict)
    allow_duplicate_hash_within_attack: bool = True


_GOAL_TARGET_POLICY = {
    # External hard-suite scoring skips non-evaluable rows.  Therefore attack
    # folders may be mixed, but each attack must contain at least one evaluable
    # row for its goal.
    "oda": "present_required",
    "positive": "present_required",
    "target_present": "present_required",
    "oga": "absent_required",
    "semantic": "mixed_allowed",
    "cleanlabel": "mixed_allowed",
    "clean-label": "mixed_allowed",
    "wanet": "absent_required",
    "blend": "absent_required",
    "badnet_oga": "absent_required",
    "fp": "absent_required",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_goal_from_name(name: str, expected_goals: Mapping[str, str] | None = None) -> str:
    low = str(name).lower()
    for key, goal in (expected_goals or {}).items():
        if str(key).lower() in low:
            return str(goal).lower()
    if "oda" in low or "vanish" in low or "disappear" in low:
        return "oda"
    if "semantic" in low or "cleanlabel" in low or "clean-label" in low:
        return "semantic"
    if "badnet_oga" in low or "oga" in low or "ghost" in low or "fp" in low:
        return "oga"
    if "wanet" in low or "blend" in low or "badnet" in low:
        return "oga"
    return "unknown"


def target_policy_for_goal(goal: str) -> str:
    g = str(goal).lower()
    if g in _GOAL_TARGET_POLICY:
        return _GOAL_TARGET_POLICY[g]
    if "oda" in g:
        return "present_required"
    if "semantic" in g:
        return "mixed_allowed"
    if "oga" in g or "wanet" in g or "blend" in g:
        return "absent_required"
    return "unknown"


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except (TypeError, ValueError):
            continue
    return classes


def find_attack_dirs(root: str | Path) -> list[tuple[str, Path, Path]]:
    root = Path(root)
    candidates: list[Path] = []
    if (root / "images").exists() and (root / "labels").exists():
        candidates.append(root)
    if (root / "data").exists():
        candidates.extend([p for p in sorted((root / "data").iterdir()) if p.is_dir()])
    candidates.extend([p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in {"data", "models", "runs", "security_gate", "label_backups"}])
    out: list[tuple[str, Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for base in candidates:
        pairs = [
            (base / "images" / "attack_eval", base / "labels" / "attack_eval"),
            (base / "images" / "val", base / "labels" / "val"),
            (base / "images" / "test", base / "labels" / "test"),
            (base / "images" / "train", base / "labels" / "train"),
            (base / "images", base / "labels"),
        ]
        for images, labels in pairs:
            if images.exists() and labels.exists():
                key = (str(images.resolve()), str(labels.resolve()))
                if key not in seen:
                    seen.add(key)
                    out.append((base.name, images, labels))
                break
    return out


def collect_records(root: str | Path, *, expected_goals: Mapping[str, str] | None = None) -> list[ImageLabelRecord]:
    records: list[ImageLabelRecord] = []
    for attack, images_dir, labels_dir in find_attack_dirs(root):
        for img in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            label = label_path_for_image(img, labels_dir)
            records.append(
                ImageLabelRecord(
                    image_path=str(img),
                    label_path=str(label),
                    attack=attack,
                    sha256=_sha256(img),
                    classes=read_label_classes(label),
                )
            )
    return records


def collect_heldout_hashes(roots: Sequence[str | Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root_like in roots or []:
        root = Path(root_like)
        if not root.exists():
            continue
        for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            hashes[_sha256(img)] = str(img)
    return hashes


def audit_benchmark(cfg: BenchmarkAuditConfig) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    all_records: list[ImageLabelRecord] = []
    for root in cfg.roots:
        root_path = Path(root)
        if not root_path.exists():
            findings.append({"severity": "error", "code": "missing_root", "path": str(root)})
            continue
        recs = collect_records(root_path, expected_goals=cfg.expected_goals)
        if not recs:
            findings.append({"severity": "error", "code": "no_attack_records", "path": str(root)})
        all_records.extend(recs)

    by_attack: dict[str, list[ImageLabelRecord]] = {}
    for rec in all_records:
        by_attack.setdefault(rec.attack, []).append(rec)

    target = int(cfg.target_class_id)
    attacks_summary = {}
    for attack, recs in sorted(by_attack.items()):
        goal = infer_goal_from_name(attack, cfg.expected_goals)
        policy = target_policy_for_goal(goal)
        target_present = sum(1 for r in recs if target in r.classes)
        missing_labels = sum(1 for r in recs if not Path(r.label_path).exists())
        attacks_summary[attack] = {
            "n_images": len(recs),
            "goal": goal,
            "target_policy": policy,
            "target_present_count": target_present,
            "target_absent_count": len(recs) - target_present,
            "missing_label_count": missing_labels,
        }
        if cfg.min_images_per_attack and len(recs) < int(cfg.min_images_per_attack):
            findings.append({"severity": "warning", "code": "low_attack_count", "attack": attack, "n": len(recs), "min": int(cfg.min_images_per_attack)})
        if policy == "present_required" and target_present == 0:
            findings.append({"severity": "error", "code": "no_evaluable_target_present_rows", "attack": attack})
        if policy == "absent_required" and target_present >= len(recs):
            findings.append({"severity": "error", "code": "no_evaluable_target_absent_rows", "attack": attack})
        if policy == "absent_required" and 0 < target_present < len(recs):
            findings.append({"severity": "warning", "code": "mixed_rows_in_target_absent_attack", "attack": attack, "target_present_rows_skipped_by_scorer": target_present})
        if policy == "present_required" and 0 < (len(recs) - target_present) < len(recs):
            findings.append({"severity": "warning", "code": "mixed_rows_in_target_present_attack", "attack": attack, "target_absent_rows_skipped_by_scorer": len(recs) - target_present})
        if missing_labels:
            findings.append({"severity": "warning", "code": "missing_label_files", "attack": attack, "n": missing_labels})

    by_hash: dict[str, list[ImageLabelRecord]] = {}
    for rec in all_records:
        by_hash.setdefault(rec.sha256, []).append(rec)
    for sha, recs in sorted(by_hash.items()):
        attacks = sorted({r.attack for r in recs})
        if len(attacks) > 1 or (len(recs) > 1 and not cfg.allow_duplicate_hash_within_attack):
            findings.append({"severity": "warning", "code": "duplicate_image_hash", "sha256": sha, "attacks": attacks, "paths": [r.image_path for r in recs[:5]]})

    heldout_hashes = collect_heldout_hashes(cfg.heldout_roots)
    overlaps = []
    for rec in all_records:
        if rec.sha256 in heldout_hashes:
            overlaps.append({"benchmark_path": rec.image_path, "heldout_path": heldout_hashes[rec.sha256], "sha256": rec.sha256, "attack": rec.attack})
    if overlaps:
        findings.append({"severity": "error", "code": "heldout_overlap", "n": len(overlaps), "examples": overlaps[:10]})

    errors = sum(1 for f in findings if f.get("severity") == "error")
    warnings = sum(1 for f in findings if f.get("severity") == "warning")
    return {
        "passed": errors == 0,
        "n_records": len(all_records),
        "n_attacks": len(by_attack),
        "attacks": attacks_summary,
        "n_errors": errors,
        "n_warnings": warnings,
        "findings": findings,
        "config": asdict(cfg),
    }


def write_audit(path: str | Path, audit: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
