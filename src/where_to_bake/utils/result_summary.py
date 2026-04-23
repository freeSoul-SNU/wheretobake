"""Aggregate result JSON files into flat summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _flatten_result(result: dict[str, Any], source_path: str) -> dict[str, Any]:
    fidelity = result.get("teacher_fidelity_metrics", {})
    preservation = result.get("preservation_metrics", {})
    efficiency = result.get("efficiency_metrics", {})
    return {
        "result_path": source_path,
        "run_name": result.get("run_name"),
        "baseline_name": result.get("baseline_name"),
        "model_name": result.get("model_name"),
        "seed": result.get("seed"),
        "prompt_family": result.get("prompt_family"),
        "paraphrase_split": result.get("paraphrase_split"),
        "trainable_params": result.get("trainable_params"),
        "train_runtime_sec": result.get("train_runtime_sec"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "token_kl": fidelity.get("token_kl"),
        "next_token_agreement": fidelity.get("next_token_agreement"),
        "style_agreement": fidelity.get("style_agreement"),
        "base_drift_kl": preservation.get("base_drift_kl"),
        "unrelated_input_drift": preservation.get("unrelated_input_drift"),
        "train_tokens_per_sec": efficiency.get("train_tokens_per_sec"),
        "eval_tokens_per_sec": efficiency.get("eval_tokens_per_sec"),
        "selection_strategy": result.get("selection_strategy"),
        "selection_budget": result.get("selection_budget"),
    }


def collect_result_files(root_dir: str | Path) -> list[Path]:
    """Collect all result JSON files under an output root."""

    root = Path(root_dir)
    return sorted(root.glob("**/result.json"))


def build_summary_rows(root_dir: str | Path) -> list[dict[str, Any]]:
    """Build a flat summary table from result JSON files."""

    rows: list[dict[str, Any]] = []
    for path in collect_result_files(root_dir):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append(_flatten_result(result, str(path)))
    return rows


def write_summary_files(rows: list[dict[str, Any]], output_prefix: str | Path) -> tuple[Path, Path]:
    """Write summary rows to JSON and CSV."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else [
        "result_path",
        "run_name",
        "baseline_name",
        "model_name",
        "seed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path

