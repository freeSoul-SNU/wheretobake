"""Generate long-form prompt-family datasets from a seed corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _approved_prompts_by_family(prompt_family_spec_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = _load_yaml(prompt_family_spec_path)
    result: dict[str, list[dict[str, Any]]] = {}
    for family in payload.get("families", []):
        family_name = family["family_name"]
        prompts = [
            prompt
            for prompt in family.get("prompts", [])
            if prompt.get("qc_status") == "approved"
        ]
        result[family_name] = prompts
    return result


def generate_longform_dataset(
    source_corpus_path: str | Path,
    prompt_family_spec_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    """Generate train/valid/test/preserve JSONL files from long-form source data."""

    source_payload = _load_yaml(source_corpus_path)
    prompts_by_family = _approved_prompts_by_family(prompt_family_spec_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
        "preserve": [],
    }

    for item in source_payload.get("examples", []):
        split = item["split"]
        source_id = item["source_id"]
        for family_name, target_text in item["family_targets"].items():
            prompts = [
                prompt
                for prompt in prompts_by_family.get(family_name, [])
                if prompt.get("split") == split
            ]
            for prompt in prompts:
                records_by_split[split].append(
                    {
                        "example_id": f"{split}_{source_id}_{prompt['prompt_id']}",
                        "source_id": source_id,
                        "prompt_family": family_name,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_text": prompt["prompt_text"],
                        "input_text": item["source_text"],
                        "target_text": target_text,
                        "split": split,
                        "paraphrase_split": prompt["paraphrase_split"],
                        "qc_status": prompt["qc_status"],
                        "notes": item["title"],
                    }
                )

    for preserve_item in source_payload.get("preserve_examples", []):
        records_by_split["preserve"].append(preserve_item)

    filename_map = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl",
        "preserve": "preserve.jsonl",
    }
    counts: dict[str, int] = {}
    for split, filename in filename_map.items():
        path = output_root / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in records_by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(records_by_split[split])
    return counts

