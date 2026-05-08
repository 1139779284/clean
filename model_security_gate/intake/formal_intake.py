"""Formal intake validation for models before security scanning.

The roadmap calls out model card, training log, preprocess, class map, artifact
hash, and provenance validation. This module turns those checks into a concrete
manifest. It is intentionally lightweight: it validates metadata and artifact
consistency without loading the model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import hashlib
import json
import os

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class FormalIntakeConfig:
    require_model_card: bool = True
    require_training_log: bool = False
    require_preprocess: bool = True
    require_class_map: bool = True
    require_provenance: bool = True
    allowed_model_suffixes: List[str] = field(default_factory=lambda: [".pt", ".onnx", ".engine"])
    required_model_card_fields: List[str] = field(
        default_factory=lambda: [
            "model_name",
            "model_version",
            "owner",
            "training_data",
            "class_names",
            "preprocess",
            "intended_use",
            "known_risks",
        ]
    )
    required_preprocess_fields: List[str] = field(
        default_factory=lambda: ["imgsz", "letterbox", "color_space", "normalization"]
    )
    min_class_count: int = 1
    expected_task: Optional[str] = "detect"
    max_artifact_bytes: Optional[int] = None


@dataclass
class FormalIntakeResult:
    accepted: bool
    blockers: List[str]
    warnings: List[str]
    manifest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_structured_file(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML intake files")
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported structured file type: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _as_list_class_names(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, tuple):
        return [str(x) for x in obj]
    if isinstance(obj, Mapping):
        names = obj.get("names", obj)
        if isinstance(names, Mapping):
            return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
        if isinstance(names, list):
            return [str(x) for x in names]
    return []


def _lookup_nested(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        cur: Any = mapping
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _validate_required_fields(label: str, data: Mapping[str, Any], fields: Iterable[str]) -> List[str]:
    missing = []
    for field in fields:
        value = _lookup_nested(data, [field])
        if value is None or value == "":
            missing.append(f"{label} missing required field: {field}")
    return missing


def build_intake_manifest(
    *,
    model_path: str | Path,
    model_card_path: str | Path | None = None,
    training_log_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    preprocess_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    config: Optional[FormalIntakeConfig] = None,
) -> FormalIntakeResult:
    cfg = config or FormalIntakeConfig()
    blockers: List[str] = []
    warnings: List[str] = []

    model_path = Path(model_path)
    artifact: Dict[str, Any] = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "suffix": model_path.suffix.lower(),
    }
    if not model_path.exists():
        blockers.append(f"model artifact not found: {model_path}")
    else:
        size_bytes = model_path.stat().st_size
        artifact.update(
            {
                "size_bytes": size_bytes,
                "sha256": sha256_file(model_path),
                "mtime_utc": datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
        if cfg.allowed_model_suffixes and model_path.suffix.lower() not in cfg.allowed_model_suffixes:
            blockers.append(f"unsupported model suffix: {model_path.suffix}")
        if cfg.max_artifact_bytes is not None and size_bytes > cfg.max_artifact_bytes:
            blockers.append(f"model artifact size {size_bytes} exceeds limit {cfg.max_artifact_bytes}")

    model_card = _read_structured_file(model_card_path) if model_card_path else {}
    data_yaml = _read_structured_file(data_yaml_path) if data_yaml_path else {}
    preprocess = _read_structured_file(preprocess_path) if preprocess_path else {}
    provenance = _read_structured_file(provenance_path) if provenance_path else {}

    if cfg.require_model_card and not model_card:
        blockers.append("model card is required but missing")
    if model_card:
        blockers.extend(_validate_required_fields("model_card", model_card, cfg.required_model_card_fields))
        task = _lookup_nested(model_card, ["task", "model.task", "metadata.task"])
        if cfg.expected_task and task and str(task).lower() != cfg.expected_task.lower():
            blockers.append(f"model_card task {task!r} does not match expected task {cfg.expected_task!r}")

    if cfg.require_training_log and not training_log_path:
        blockers.append("training log is required but missing")
    if training_log_path:
        p = Path(training_log_path)
        if not p.exists():
            blockers.append(f"training log not found: {p}")

    if cfg.require_preprocess and not preprocess:
        blockers.append("preprocess spec is required but missing")
    if preprocess:
        blockers.extend(_validate_required_fields("preprocess", preprocess, cfg.required_preprocess_fields))

    class_names = []
    class_source = "none"
    if data_yaml:
        class_names = _as_list_class_names(data_yaml)
        class_source = "data_yaml"
    if not class_names and model_card:
        class_names = _as_list_class_names(model_card.get("class_names"))
        class_source = "model_card"
    if cfg.require_class_map and len(class_names) < cfg.min_class_count:
        blockers.append("class map is required but no class names were found")
    if len(set(class_names)) != len(class_names):
        blockers.append("class map contains duplicate class names")

    if cfg.require_provenance:
        provenance_ok = bool(provenance) or bool(_lookup_nested(model_card, ["provenance", "training_data", "owner"]))
        if not provenance_ok:
            blockers.append("provenance is required but neither provenance file nor model-card provenance fields are present")
    if provenance:
        source = _lookup_nested(provenance, ["source", "artifact.source", "model.source"])
        if not source:
            warnings.append("provenance file has no source field")

    manifest = {
        "schema_version": "model-security-gate-intake/v1",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "artifact": artifact,
        "model_card_path": str(model_card_path) if model_card_path else None,
        "training_log_path": str(training_log_path) if training_log_path else None,
        "data_yaml_path": str(data_yaml_path) if data_yaml_path else None,
        "preprocess_path": str(preprocess_path) if preprocess_path else None,
        "provenance_path": str(provenance_path) if provenance_path else None,
        "class_map": {"source": class_source, "count": len(class_names), "names": class_names},
        "preprocess": preprocess,
        "model_card_summary": {
            "model_name": model_card.get("model_name"),
            "model_version": model_card.get("model_version"),
            "owner": model_card.get("owner"),
            "task": model_card.get("task"),
            "intended_use": model_card.get("intended_use"),
        },
        "provenance": provenance,
        "config": asdict(cfg),
    }
    return FormalIntakeResult(accepted=not blockers, blockers=blockers, warnings=warnings, manifest=manifest)


def run_formal_intake(
    *,
    model_path: str | Path,
    output_path: str | Path | None = None,
    model_card_path: str | Path | None = None,
    training_log_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    preprocess_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    config: Optional[FormalIntakeConfig] = None,
) -> FormalIntakeResult:
    result = build_intake_manifest(
        model_path=model_path,
        model_card_path=model_card_path,
        training_log_path=training_log_path,
        data_yaml_path=data_yaml_path,
        preprocess_path=preprocess_path,
        provenance_path=provenance_path,
        config=config,
    )
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def load_intake_config(path: str | Path | None) -> FormalIntakeConfig:
    if not path:
        return FormalIntakeConfig()
    data = _read_structured_file(path)
    if "intake" in data and isinstance(data["intake"], Mapping):
        data = dict(data["intake"])
    return FormalIntakeConfig(**data)
