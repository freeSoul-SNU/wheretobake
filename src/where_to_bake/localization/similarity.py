"""Prompt-family similarity inspection based on prompt-induced module deltas."""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from where_to_bake.baselines.selection import list_lora_candidates
from where_to_bake.data import filter_records, load_jsonl_records
from where_to_bake.data.prompt_dataset import format_student_prompt, format_teacher_prompt
from where_to_bake.models.wrapper import _require_hf_stack, load_base_models, load_tokenizer
from where_to_bake.utils.io import save_json


@dataclass
class DeltaExample:
    """Prompt-induced delta for one prompt on one source input."""

    source_id: str
    prompt_family: str
    prompt_id: str
    prompt_text: str
    split: str
    paraphrase_split: str
    module_deltas: dict[str, Any]


def _load_localization_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_config = config["data"]
    localization_config = config.get("localization", {})
    split_to_path = {
        "train": data_config["train_path"],
        "valid": data_config["valid_path"],
        "test": data_config["test_path"],
    }
    requested_splits = localization_config.get("splits", ["train", "valid", "test"])
    paraphrase_split = localization_config.get("paraphrase_split", "all")
    family_scope = config["baseline"].get("family_scope", "all")

    records: list[dict[str, Any]] = []
    for split in requested_splits:
        split_records = load_jsonl_records(split_to_path[split])
        split_records = filter_records(
            records=split_records,
            family_scope=family_scope,
            paraphrase_split=paraphrase_split,
        )
        records.extend(split_records)

    max_records = localization_config.get("max_records")
    if max_records:
        records = records[:max_records]
    return records


def _summarize_tensor(output: Any, torch: Any) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    tensor = tensor.detach()
    if tensor.dim() == 3:
        return tensor.mean(dim=1).squeeze(0).float().cpu()
    if tensor.dim() == 2:
        return tensor.mean(dim=0).float().cpu()
    return tensor.reshape(-1).float().cpu()


def _register_summary_hooks(model: Any, module_names: list[str], torch: Any) -> tuple[dict[str, Any], list[Any]]:
    store: dict[str, Any] = {}
    module_map = dict(model.named_modules())
    handles = []
    for module_name in module_names:
        module = module_map[module_name]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, name: str = module_name) -> None:
            store[name] = _summarize_tensor(output, torch)

        handles.append(module.register_forward_hook(hook))
    return store, handles


def _run_forward_with_hooks(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int,
    device: Any,
    module_names: list[str],
    torch: Any,
) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    store, handles = _register_summary_hooks(model, module_names, torch)
    try:
        with torch.no_grad():
            model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return store


def collect_prompt_deltas(config: dict[str, Any]) -> tuple[list[DeltaExample], list[str]]:
    """Collect prompt-induced module deltas for similarity inspection."""

    torch, AutoModelForCausalLM, AutoTokenizer, _peft_items = _require_hf_stack()
    tokenizer = load_tokenizer(config, AutoTokenizer)
    teacher_model, base_model = load_base_models(config, AutoModelForCausalLM)
    device = torch.device(config["run"]["device"])
    teacher_model.to(device)
    base_model.to(device)
    teacher_model.eval()
    base_model.eval()

    localization_config = config.get("localization", {})
    suffixes = localization_config.get(
        "candidate_module_suffixes",
        config.get("selection", {}).get("candidate_module_suffixes", config["lora"]["target_modules"]),
    )
    candidate_names = list(list_lora_candidates(base_model, suffixes).keys())
    if not candidate_names:
        raise ValueError(f"No candidate modules found for localization with suffixes: {suffixes}")

    max_length = config["data"]["max_source_length"]
    examples: list[DeltaExample] = []
    for row in _load_localization_records(config):
        source_id = row.get("source_id") or row.get("example_id") or row[config["prompting"]["input_field"]]
        prompt_text = row[config["prompting"]["prompt_field"]]
        input_text = row[config["prompting"]["input_field"]]

        teacher_text = format_teacher_prompt(prompt_text, input_text)
        base_text = format_student_prompt(input_text)

        teacher_store = _run_forward_with_hooks(
            model=teacher_model,
            tokenizer=tokenizer,
            text=teacher_text,
            max_length=max_length,
            device=device,
            module_names=candidate_names,
            torch=torch,
        )
        base_store = _run_forward_with_hooks(
            model=base_model,
            tokenizer=tokenizer,
            text=base_text,
            max_length=max_length,
            device=device,
            module_names=candidate_names,
            torch=torch,
        )
        module_deltas = {
            name: teacher_store[name] - base_store[name]
            for name in candidate_names
            if name in teacher_store and name in base_store
        }
        examples.append(
            DeltaExample(
                source_id=str(source_id),
                prompt_family=row.get("prompt_family", "all"),
                prompt_id=row.get("prompt_id", row.get("example_id", "unknown")),
                prompt_text=prompt_text,
                split=row.get("split", "unknown"),
                paraphrase_split=row.get("paraphrase_split", "all"),
                module_deltas=module_deltas,
            )
        )
    return examples, candidate_names


def _cosine_similarity(left: Any, right: Any, torch: Any) -> float:
    left_norm = left.norm(p=2).item()
    right_norm = right.norm(p=2).item()
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _mean_and_count(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "count": 0}
    return {"mean": sum(values) / len(values), "count": len(values)}


def compute_similarity_report(
    examples: list[DeltaExample],
    module_names: list[str],
    alpha: float,
) -> dict[str, Any]:
    """Compute within-family and across-family similarity from collected deltas."""

    import torch

    grouped_by_source: dict[str, list[DeltaExample]] = {}
    for example in examples:
        grouped_by_source.setdefault(example.source_id, []).append(example)

    families = sorted({example.prompt_family for example in examples})
    within_scores: dict[str, dict[str, list[float]]] = {
        family: {module_name: [] for module_name in module_names} for family in families
    }
    across_scores: dict[str, dict[str, list[float]]] = {
        family: {module_name: [] for module_name in module_names} for family in families
    }
    pair_samples: list[dict[str, Any]] = []

    for source_examples in grouped_by_source.values():
        for left, right in itertools.combinations(source_examples, 2):
            shared_modules = set(left.module_deltas).intersection(right.module_deltas)
            if not shared_modules:
                continue
            pair_type = "within" if left.prompt_family == right.prompt_family else "across"
            sample_payload = {
                "source_id": left.source_id,
                "pair_type": pair_type,
                "left_family": left.prompt_family,
                "right_family": right.prompt_family,
                "left_prompt_id": left.prompt_id,
                "right_prompt_id": right.prompt_id,
                "module_similarities": {},
            }
            for module_name in sorted(shared_modules):
                similarity = _cosine_similarity(
                    left.module_deltas[module_name],
                    right.module_deltas[module_name],
                    torch,
                )
                sample_payload["module_similarities"][module_name] = similarity
                if pair_type == "within":
                    within_scores[left.prompt_family][module_name].append(similarity)
                else:
                    across_scores[left.prompt_family][module_name].append(similarity)
                    across_scores[right.prompt_family][module_name].append(similarity)
            if len(pair_samples) < 20:
                pair_samples.append(sample_payload)

    within_summary = {
        family: {
            module_name: _mean_and_count(values)
            for module_name, values in modules.items()
        }
        for family, modules in within_scores.items()
    }
    across_summary = {
        family: {
            module_name: _mean_and_count(values)
            for module_name, values in modules.items()
        }
        for family, modules in across_scores.items()
    }

    stability_summary: dict[str, dict[str, Any]] = {}
    for family in families:
        module_stats: dict[str, Any] = {}
        for module_name in module_names:
            within_mean = within_summary[family][module_name]["mean"]
            across_mean = across_summary[family][module_name]["mean"]
            if within_mean is None:
                score = None
            else:
                score = within_mean - alpha * (across_mean or 0.0)
            module_stats[module_name] = {
                "within_family_consistency": within_mean,
                "across_family_similarity": across_mean,
                "stability_score": score,
                "within_count": within_summary[family][module_name]["count"],
                "across_count": across_summary[family][module_name]["count"],
            }
        stability_summary[family] = module_stats

    top_modules_by_family = {}
    for family, modules in stability_summary.items():
        ranked = sorted(
            (
                {"module_name": module_name, **stats}
                for module_name, stats in modules.items()
                if stats["stability_score"] is not None
            ),
            key=lambda row: row["stability_score"],
            reverse=True,
        )
        top_modules_by_family[family] = ranked[:10]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "alpha": alpha,
        "example_count": len(examples),
        "family_names": families,
        "candidate_modules": module_names,
        "within_family_summary": within_summary,
        "across_family_summary": across_summary,
        "stability_summary": stability_summary,
        "top_modules_by_family": top_modules_by_family,
        "pair_samples": pair_samples,
    }


def save_similarity_report(report: dict[str, Any], output_prefix: str | Path) -> tuple[Path, Path]:
    """Save similarity report as JSON and CSV."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    save_json(json_path, report)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "module_name",
                "within_family_consistency",
                "across_family_similarity",
                "stability_score",
                "within_count",
                "across_count",
            ],
        )
        writer.writeheader()
        for family, module_stats in report["stability_summary"].items():
            for module_name, stats in module_stats.items():
                writer.writerow(
                    {
                        "family": family,
                        "module_name": module_name,
                        **stats,
                    }
                )
    return json_path, csv_path

