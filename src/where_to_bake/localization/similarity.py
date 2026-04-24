"""Prompt-family localization utilities based on prompt-induced module deltas."""

from __future__ import annotations

import csv
import itertools
import math
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
    """Prompt-induced localization signal for one prompt on one source input."""

    source_id: str
    prompt_family: str
    prompt_id: str
    prompt_text: str
    split: str
    paraphrase_split: str
    module_deltas: dict[str, Any]
    causal_effects: dict[str, float] | None = None


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


def _resolve_candidate_module_names(model: Any, config: dict[str, Any]) -> list[str]:
    localization_config = config.get("localization", {})
    suffixes = localization_config.get(
        "candidate_module_suffixes",
        config.get("selection", {}).get("candidate_module_suffixes", config["lora"]["target_modules"]),
    )
    candidate_names = list(list_lora_candidates(model, suffixes).keys())
    if not candidate_names:
        raise ValueError(f"No candidate modules found for localization with suffixes: {suffixes}")
    return candidate_names


def _build_localization_sequence(
    tokenizer: Any,
    prefix_text: str,
    target_text: str,
    max_source_length: int,
    max_target_length: int,
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], Any]:
    eos_text = tokenizer.eos_token or ""
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_length,
    )["input_ids"]
    target_piece = f" {target_text}{eos_text}"
    target_ids = tokenizer(
        target_piece,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )["input_ids"]
    input_ids = prefix_ids + target_ids
    attention_mask = [1] * len(input_ids)
    response_mask = [0] * len(prefix_ids) + [1] * len(target_ids)
    encoded = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
    }
    return encoded, torch.tensor(response_mask, dtype=torch.bool, device=device)


def _extract_response_hidden(output: Any, response_mask: Any, torch: Any) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    tensor = tensor.detach()
    if tensor.dim() != 3:
        return tensor.reshape(1, -1).float().cpu()
    return tensor[0, response_mask.to(device=tensor.device), :].float().cpu()


def _pool_response_hidden(
    response_hidden: Any,
    strategy: str,
    last_k: int,
    torch: Any,
) -> Any:
    if response_hidden.numel() == 0:
        width = response_hidden.shape[-1] if response_hidden.dim() >= 2 else 1
        if strategy == "response_last_k_concat":
            return torch.zeros(width * last_k, dtype=torch.float32)
        return torch.zeros(width, dtype=torch.float32)

    if strategy == "response_mean":
        return response_hidden.mean(dim=0)
    if strategy == "response_last":
        return response_hidden[-1]
    if strategy == "response_last_k_mean":
        return response_hidden[-last_k:].mean(dim=0)
    if strategy == "response_last_k_concat":
        hidden_size = response_hidden.shape[-1]
        selected = response_hidden[-last_k:]
        if selected.shape[0] < last_k:
            pad = torch.zeros(last_k - selected.shape[0], hidden_size, dtype=selected.dtype)
            selected = torch.cat([pad, selected], dim=0)
        return selected.reshape(-1)
    raise ValueError(f"Unsupported localization.representation_pooling: {strategy}")


def _register_summary_hooks(
    model: Any,
    module_names: list[str],
    response_mask: Any,
    pool_strategy: str,
    last_k: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    pooled_store: dict[str, Any] = {}
    response_hidden_store: dict[str, Any] = {}
    module_map = dict(model.named_modules())
    handles = []
    for module_name in module_names:
        module = module_map[module_name]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, name: str = module_name) -> None:
            response_hidden = _extract_response_hidden(output, response_mask, torch)
            response_hidden_store[name] = response_hidden
            pooled_store[name] = _pool_response_hidden(
                response_hidden=response_hidden,
                strategy=pool_strategy,
                last_k=last_k,
                torch=torch,
            )

        handles.append(module.register_forward_hook(hook))
    return pooled_store, response_hidden_store, handles


def _run_forward_with_hooks(
    model: Any,
    encoded: Any,
    module_names: list[str],
    response_mask: Any,
    pool_strategy: str,
    last_k: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    pooled_store, response_hidden_store, handles = _register_summary_hooks(
        model=model,
        module_names=module_names,
        response_mask=response_mask,
        pool_strategy=pool_strategy,
        last_k=last_k,
        torch=torch,
    )
    try:
        with torch.no_grad():
            outputs = model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return pooled_store, response_hidden_store, outputs.logits.detach().float().cpu()


def _apply_delta_ablation(output: Any, response_mask: Any, delta_hidden: Any) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    response_mask = response_mask.to(device=tensor.device)
    delta_hidden = delta_hidden.to(device=tensor.device, dtype=tensor.dtype)
    if tensor.dim() == 3:
        adjusted = tensor.clone()
        adjusted[0, response_mask, :] = adjusted[0, response_mask, :] - delta_hidden
    elif tensor.dim() == 2:
        adjusted = tensor.clone()
        adjusted[response_mask, :] = adjusted[response_mask, :] - delta_hidden
    else:
        adjusted = tensor - delta_hidden.reshape_as(tensor)
    if isinstance(output, tuple):
        return (adjusted, *output[1:])
    return adjusted


def _run_ablation_forward(
    model: Any,
    encoded: Any,
    module_name: str,
    response_mask: Any,
    delta_hidden: Any,
    torch: Any,
) -> Any:
    module = dict(model.named_modules())[module_name]

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return _apply_delta_ablation(output, response_mask, delta_hidden)

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(**encoded)
    finally:
        handle.remove()
    return outputs.logits.detach().float().cpu()


def _response_region_kl(reference_logits: Any, perturbed_logits: Any, response_mask: Any, torch: Any) -> float:
    response_mask = response_mask.to(dtype=torch.bool, device=reference_logits.device)
    reference_slice = reference_logits[:, response_mask, :]
    perturbed_slice = perturbed_logits[:, response_mask, :]
    if reference_slice.numel() == 0:
        return 0.0
    reference_probs = torch.softmax(reference_slice, dim=-1)
    perturbed_log_probs = torch.log_softmax(perturbed_slice, dim=-1)
    token_kl = torch.nn.functional.kl_div(perturbed_log_probs, reference_probs, reduction="none").sum(dim=-1)
    return float(token_kl.mean().item())


def collect_prompt_deltas(config: dict[str, Any]) -> tuple[list[DeltaExample], list[str]]:
    """Collect prompt-induced deltas and optional sequence-level causal effects."""

    torch, AutoModelForCausalLM, AutoTokenizer, _peft_items = _require_hf_stack()
    tokenizer = load_tokenizer(config, AutoTokenizer)
    teacher_model, base_model = load_base_models(config, AutoModelForCausalLM)
    device = torch.device(config["run"]["device"])
    teacher_model.to(device)
    base_model.to(device)
    teacher_model.eval()
    base_model.eval()

    candidate_names = _resolve_candidate_module_names(base_model, config)
    max_source_length = config["data"]["max_source_length"]
    max_target_length = config["data"]["max_target_length"]
    localization_config = config.get("localization", {})
    compute_causal = localization_config.get("compute_causal", False)
    pool_strategy = localization_config.get("representation_pooling", "response_last_k_concat")
    last_k = localization_config.get("response_last_k", 4)

    examples: list[DeltaExample] = []
    for row in _load_localization_records(config):
        source_id = row.get("source_id") or row.get("example_id") or row[config["prompting"]["input_field"]]
        prompt_text = row[config["prompting"]["prompt_field"]]
        input_text = row[config["prompting"]["input_field"]]
        target_text = row.get(config["prompting"]["target_field"], "")

        teacher_prefix = format_teacher_prompt(prompt_text, input_text)
        base_prefix = format_student_prompt(input_text)
        teacher_encoded, teacher_response_mask = _build_localization_sequence(
            tokenizer=tokenizer,
            prefix_text=teacher_prefix,
            target_text=target_text,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            device=device,
            torch=torch,
        )
        base_encoded, base_response_mask = _build_localization_sequence(
            tokenizer=tokenizer,
            prefix_text=base_prefix,
            target_text=target_text,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            device=device,
            torch=torch,
        )

        teacher_store, teacher_response_hidden, teacher_logits = _run_forward_with_hooks(
            model=teacher_model,
            encoded=teacher_encoded,
            module_names=candidate_names,
            response_mask=teacher_response_mask,
            pool_strategy=pool_strategy,
            last_k=last_k,
            torch=torch,
        )
        base_store, base_response_hidden, _base_logits = _run_forward_with_hooks(
            model=base_model,
            encoded=base_encoded,
            module_names=candidate_names,
            response_mask=base_response_mask,
            pool_strategy=pool_strategy,
            last_k=last_k,
            torch=torch,
        )
        module_deltas = {
            name: teacher_store[name] - base_store[name]
            for name in candidate_names
            if name in teacher_store and name in base_store
        }

        causal_effects: dict[str, float] | None = None
        if compute_causal:
            causal_effects = {}
            for module_name in candidate_names:
                if module_name not in teacher_response_hidden or module_name not in base_response_hidden:
                    continue
                delta_hidden = teacher_response_hidden[module_name] - base_response_hidden[module_name]
                ablated_logits = _run_ablation_forward(
                    model=teacher_model,
                    encoded=teacher_encoded,
                    module_name=module_name,
                    response_mask=teacher_response_mask,
                    delta_hidden=delta_hidden,
                    torch=torch,
                )
                causal_effects[module_name] = _response_region_kl(
                    reference_logits=teacher_logits,
                    perturbed_logits=ablated_logits,
                    response_mask=teacher_response_mask.cpu(),
                    torch=torch,
                )

        examples.append(
            DeltaExample(
                source_id=str(source_id),
                prompt_family=row.get("prompt_family", "all"),
                prompt_id=row.get("prompt_id", row.get("example_id", "unknown")),
                prompt_text=prompt_text,
                split=row.get("split", "unknown"),
                paraphrase_split=row.get("paraphrase_split", "all"),
                module_deltas=module_deltas,
                causal_effects=causal_effects,
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


def _zscore(values: dict[str, float | None], transform: str = "identity") -> dict[str, float | None]:
    transformed: dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        if transform == "log1p":
            transformed[key] = math.log1p(max(value, 0.0))
        else:
            transformed[key] = value
    if not transformed:
        return {key: None for key in values}
    mean_value = sum(transformed.values()) / len(transformed)
    variance = sum((value - mean_value) ** 2 for value in transformed.values()) / len(transformed)
    std_value = variance**0.5
    if std_value == 0.0:
        return {key: 0.0 if key in transformed else None for key in values}
    return {
        key: ((transformed[key] - mean_value) / std_value) if key in transformed else None
        for key in values
    }


def compute_similarity_report(
    examples: list[DeltaExample],
    module_names: list[str],
    alpha: float,
    pooling_strategy: str = "response_last_k_concat",
    response_last_k: int = 4,
    causal_metric: str = "response_region_kl",
) -> dict[str, Any]:
    """Compute within/across similarity and sequence-level causal summaries."""

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
    causal_scores: dict[str, dict[str, list[float]]] = {
        family: {module_name: [] for module_name in module_names} for family in families
    }
    pair_samples: list[dict[str, Any]] = []

    for example in examples:
        if example.causal_effects:
            for module_name, score in example.causal_effects.items():
                causal_scores[example.prompt_family][module_name].append(score)

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
    causal_summary = {
        family: {
            module_name: _mean_and_count(values)
            for module_name, values in modules.items()
        }
        for family, modules in causal_scores.items()
    }

    localization_summary: dict[str, dict[str, Any]] = {}
    for family in families:
        module_stats: dict[str, Any] = {}
        for module_name in module_names:
            within_mean = within_summary[family][module_name]["mean"]
            across_mean = across_summary[family][module_name]["mean"]
            causal_mean = causal_summary[family][module_name]["mean"]
            family_specificity_gap = None if within_mean is None else within_mean - (across_mean or 0.0)
            stability_score = None if within_mean is None else within_mean - alpha * (across_mean or 0.0)
            combined_score = None if stability_score is None or causal_mean is None else stability_score * causal_mean
            module_stats[module_name] = {
                "within_family_consistency": within_mean,
                "across_family_similarity": across_mean,
                "family_specificity_gap": family_specificity_gap,
                "stability_score": stability_score,
                "causal_score": causal_mean,
                "combined_score": combined_score,
                "within_count": within_summary[family][module_name]["count"],
                "across_count": across_summary[family][module_name]["count"],
                "causal_count": causal_summary[family][module_name]["count"],
            }

        stability_z = _zscore(
            {module_name: stats["stability_score"] for module_name, stats in module_stats.items()},
            transform="identity",
        )
        causal_z = _zscore(
            {module_name: stats["causal_score"] for module_name, stats in module_stats.items()},
            transform="log1p",
        )
        for module_name, stats in module_stats.items():
            stats["normalized_stability_score"] = stability_z[module_name]
            stats["normalized_log_causal_score"] = causal_z[module_name]
            if stability_z[module_name] is None or causal_z[module_name] is None:
                stats["selection_score"] = None
            else:
                stats["selection_score"] = stability_z[module_name] + causal_z[module_name]
        localization_summary[family] = module_stats

    top_modules_by_family: dict[str, list[dict[str, Any]]] = {}
    for family, modules in localization_summary.items():
        ranked = sorted(
            (
                {"module_name": module_name, **stats}
                for module_name, stats in modules.items()
                if stats["stability_score"] is not None
            ),
            key=lambda row: (
                row["selection_score"] if row["selection_score"] is not None else float("-inf"),
                row["stability_score"] if row["stability_score"] is not None else float("-inf"),
                row["causal_score"] if row["causal_score"] is not None else float("-inf"),
            ),
            reverse=True,
        )
        top_modules_by_family[family] = ranked[:10]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "alpha": alpha,
        "example_count": len(examples),
        "family_names": families,
        "candidate_modules": module_names,
        "pooling_strategy": pooling_strategy,
        "response_last_k": response_last_k,
        "causal_metric": causal_metric,
        "within_family_summary": within_summary,
        "across_family_summary": across_summary,
        "causal_summary": causal_summary,
        "stability_summary": localization_summary,
        "top_modules_by_family": top_modules_by_family,
        "pair_samples": pair_samples,
    }


def save_similarity_report(report: dict[str, Any], output_prefix: str | Path) -> tuple[Path, Path]:
    """Save localization report as JSON and CSV."""

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
                "family_specificity_gap",
                "stability_score",
                "causal_score",
                "combined_score",
                "normalized_stability_score",
                "normalized_log_causal_score",
                "selection_score",
                "within_count",
                "across_count",
                "causal_count",
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
