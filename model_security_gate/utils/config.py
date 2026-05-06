from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, TypeVar

import yaml

T = TypeVar("T")


def load_yaml_config(path: str | Path | None, section: str | None = None) -> dict[str, Any]:
    """Load a YAML config file and optionally return one section.

    Missing paths are treated as empty configs so scripts can expose a uniform
    ``--config`` argument without forcing users to create files.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping/object: {path}")
    data = dict(data)
    if section and isinstance(data.get(section), Mapping):
        return dict(data[section])
    return data


def deep_merge(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge dictionaries; override values win when not None."""
    out: dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def namespace_overrides(args: argparse.Namespace, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Return argparse values that were actually set by the user.

    This expects optional arguments to use ``default=None``. Values left as None
    do not override YAML/default config values.
    """
    exclude = set(exclude or set())
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in exclude or value is None:
            continue
        out[key] = value
    return out


def dataclass_defaults(cls: type[T]) -> dict[str, Any]:
    """Return default field values from a dataclass type or instance."""
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type/instance, got {cls!r}")
    obj = cls() if isinstance(cls, type) else cls
    return asdict(obj)


def dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type/instance, got {cls!r}")
    return {f.name for f in fields(cls)}


def split_known_keys(data: Mapping[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    known_data: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in dict(data).items():
        (known_data if key in known else extra)[key] = value
    return known_data, extra


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_resolved_config(path: str | Path, config: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(config)), f, ensure_ascii=False, indent=2)
