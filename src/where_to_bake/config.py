"""Config loading and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = [
    "run",
    "model",
    "data",
    "prompting",
    "baseline",
    "lora",
    "train",
    "eval",
    "logging",
    "output",
]


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries."""

    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(path: str | Path, validate: bool = True) -> dict[str, Any]:
    """Load and resolve a YAML config with optional defaults."""

    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    merged: dict[str, Any] = {}
    for default_path in raw.pop("defaults", []):
        merged = deep_merge_dicts(
            merged,
            load_config(config_path.parent / default_path, validate=False),
        )
    merged = deep_merge_dicts(merged, raw)
    merged["config_path"] = str(config_path)
    if validate:
        validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level config keys."""

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required config sections: {missing_text}")
