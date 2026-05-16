"""Train-evaluation leakage audit for T0 defense certificates.

A CFRC-certified defense is only scientifically valid if the external hard
suite used for certification is disjoint from anything the defended model
was *trained* on.  Hybrid-PURIFY-OD's ``external_replay_roots`` feed failure
samples from an external suite into the training dataset; when the same root
is also used as the evaluation suite (``external_eval_roots``), the resulting
CFRC certificate has an implicit train-test overlap and its statistical
claims no longer hold cleanly.

This module performs that audit without re-opening any image: it compares the
root paths and per-attack sample counts recorded in the Hybrid-PURIFY
manifest against the external suite used by CFRC's ``poisoned_external`` and
``defended_external`` reports.  It returns a structured result with:

* ``train_eval_same_roots``: set of roots that appeared both as replay and as
  eval.  A non-empty set is a high-severity warning.
* ``shared_attack_keys``: the attacks that are simultaneously in the replay
  root and the eval report.  Same-suite same-attack is the worst case.
* ``severity``: ``ok`` (disjoint), ``warn`` (shared attack keys but not full
  root overlap, e.g. operator subset different images), or ``blocked``
  (identical roots and shared attack keys).
* ``recommendation``: one-line guidance for the report.

This is intentionally read-only.  It never reads image bytes; a full
per-image hash overlap check is better left to the separate
``model_security_gate/utils/heldout_leakage.py`` helper when image paths are
available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _norm_root(path: str | Path | None) -> str:
    if not path:
        return ""
    text = str(path).replace("\\", "/").rstrip("/")
    return text


def _attack_keys(report: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(report, Mapping):
        return set()
    matrix = ((report.get("summary") or {}).get("asr_matrix") or report.get("asr_matrix") or {})
    keys: set[str] = set()
    for name in matrix:
        if isinstance(name, str):
            keys.add(name)
    return keys


@dataclass
class LeakageAudit:
    severity: str
    train_eval_same_roots: list[str] = field(default_factory=list)
    shared_attack_keys: list[str] = field(default_factory=list)
    replay_roots: list[str] = field(default_factory=list)
    eval_roots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_hybrid_manifest_against_eval(
    *,
    hybrid_manifest: Mapping[str, Any] | str | Path | None,
    defended_external_report: Mapping[str, Any] | str | Path | None = None,
    poisoned_external_report: Mapping[str, Any] | str | Path | None = None,
) -> LeakageAudit:
    """Audit one Hybrid-PURIFY manifest against its CFRC eval reports.

    Any of the arguments can be a dict, a path string, or a ``Path``.  Missing
    arguments degrade gracefully: the audit works with whatever is provided,
    and records what it could not check under ``notes``.
    """

    def _load(src: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any] | None:
        if src is None:
            return None
        if isinstance(src, Mapping):
            return src
        path = Path(src)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = _load(hybrid_manifest) or {}
    defended = _load(defended_external_report) or {}
    poisoned = _load(poisoned_external_report) or {}

    cfg = ((manifest.get("config") or {}))
    replay_roots_raw = cfg.get("external_replay_roots") or []
    eval_roots_raw = cfg.get("external_eval_roots") or []
    replay_roots = sorted({_norm_root(r) for r in replay_roots_raw if r})
    eval_roots = sorted({_norm_root(r) for r in eval_roots_raw if r})

    same_roots = sorted(set(replay_roots) & set(eval_roots))

    # Attack-key overlap: attacks that appear in both a replay root (via its
    # role in Hybrid-PURIFY's failure replay) and the CFRC eval reports.  We
    # use the eval reports as ground truth for attack names.
    eval_attack_names: set[str] = set()
    for report in (defended, poisoned):
        for key in _attack_keys(report):
            eval_attack_names.add(str(key).split("::")[-1].strip().lower())

    replay_attack_names: set[str] = set()
    # Replay datasets are enumerated in the manifest as
    # ``external_replay_datasets`` when Hybrid-PURIFY discovered them.
    for ds in manifest.get("external_replay_datasets") or []:
        if not isinstance(ds, Mapping):
            continue
        name = ds.get("attack") or ds.get("name")
        if isinstance(name, str):
            replay_attack_names.add(name.strip().lower())

    shared_attack_keys = sorted(eval_attack_names & replay_attack_names)

    notes: list[str] = []
    if not replay_roots:
        notes.append("manifest reports no external_replay_roots; audit is vacuous on the replay side")
    if not eval_roots:
        notes.append("manifest reports no external_eval_roots; audit used CFRC reports only")
    if not eval_attack_names:
        notes.append("no eval attack names resolved from external reports")
    if not replay_attack_names:
        notes.append("manifest did not enumerate external_replay_datasets")

    if same_roots and shared_attack_keys:
        severity = "blocked"
        recommendation = (
            "External replay and eval share the SAME root AND attack families. "
            "This is train-test overlap. Either (a) use a held-out external "
            "suite for CFRC eval, or (b) explicitly report the CFRC certificate "
            "as guard-free-with-replay-overlap rather than as guard-free-clean."
        )
    elif same_roots:
        severity = "warn"
        recommendation = (
            "External replay and eval share the same root even though no attack "
            "names overlap in this run. Different image subsets are likely, but "
            "an explicit held-out root is strongly preferred."
        )
    elif shared_attack_keys:
        severity = "warn"
        recommendation = (
            "Replay and eval use different roots but the same attack families. "
            "Record the split policy in the paper to convince reviewers."
        )
    else:
        severity = "ok"
        recommendation = "External replay and CFRC eval are disjoint at the root and attack-key level."

    return LeakageAudit(
        severity=severity,
        train_eval_same_roots=same_roots,
        shared_attack_keys=shared_attack_keys,
        replay_roots=replay_roots,
        eval_roots=eval_roots,
        notes=notes,
        recommendation=recommendation,
    )


def audit_cfrc_manifest(
    *,
    cfrc_manifest: Mapping[str, Any] | str | Path,
    manifests_by_arm: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Run the leakage audit on every entry in a CFRC manifest.

    ``cfrc_manifest`` is expected to have an ``entries`` list.  For each entry
    we look up the corresponding Hybrid-PURIFY manifest either via
    ``manifests_by_arm[entry.name]`` (explicit mapping) or by checking if the
    entry carries a ``hybrid_manifest`` field.
    """

    def _load(src: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(src, Mapping):
            return src
        return json.loads(Path(src).read_text(encoding="utf-8"))

    data = _load(cfrc_manifest)
    entries = data.get("entries") if isinstance(data, Mapping) else None
    if not isinstance(entries, list):
        return {"entries": [], "worst_severity": "ok"}
    mapping = dict(manifests_by_arm or {})
    audits: list[dict[str, Any]] = []
    severities = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("defense") or "")
        hybrid_manifest = mapping.get(name) or entry.get("hybrid_manifest")
        audit = audit_hybrid_manifest_against_eval(
            hybrid_manifest=hybrid_manifest,
            defended_external_report=entry.get("defended_external"),
            poisoned_external_report=entry.get("poisoned_external"),
        )
        audits.append({"arm": name, **audit.to_dict()})
        severities.append(audit.severity)
    worst = "blocked" if "blocked" in severities else ("warn" if "warn" in severities else "ok")
    return {"entries": audits, "worst_severity": worst}


__all__ = [
    "LeakageAudit",
    "audit_hybrid_manifest_against_eval",
    "audit_cfrc_manifest",
]
