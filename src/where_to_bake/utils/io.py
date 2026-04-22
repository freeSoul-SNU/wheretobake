"""I/O helpers for configs and result artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from where_to_bake.baselines import BASELINE_REGISTRY


REQUIRED_RESULT_KEYS = [
    "run_name",
    "timestamp",
    "git_commit",
    "baseline_name",
    "model_name",
    "seed",
    "prompt_family",
    "paraphrase_split",
    "trainable_params",
    "train_runtime_sec",
    "peak_memory_mb",
    "teacher_fidelity_metrics",
    "preservation_metrics",
    "efficiency_metrics",
    "config_path",
    "resolved_config_path",
    "notes",
]


def save_json(path: str | Path, payload: Any) -> None:
    """Save JSON payload with UTF-8 encoding."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Persist resolved config snapshot."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def _validate_numeric_values(node: Any, prefix: str = "root") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _validate_numeric_values(value, f"{prefix}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _validate_numeric_values(value, f"{prefix}[{index}]")
    elif isinstance(node, float) and (math.isnan(node) or math.isinf(node)):
        raise ValueError(f"Invalid numeric value at {prefix}: {node}")


def validate_result_schema(result: dict[str, Any]) -> None:
    """Validate minimal result schema contract from docs/RESULT_SCHEMA.md."""

    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(f"Missing required result keys: {', '.join(missing)}")
    if result["baseline_name"] not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline name in result: {result['baseline_name']}")
    for key in ["teacher_fidelity_metrics", "preservation_metrics", "efficiency_metrics"]:
        if not isinstance(result[key], dict):
            raise ValueError(f"Result field '{key}' must be a dict.")
    if not result["config_path"] or not result["resolved_config_path"]:
        raise ValueError("config_path and resolved_config_path must be non-empty.")
    _validate_numeric_values(result)

