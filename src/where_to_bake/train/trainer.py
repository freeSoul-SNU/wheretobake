"""Training entrypoint for the promptbake_kl baseline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from where_to_bake.baselines.selection import resolve_baseline_selection
from where_to_bake.data import (
    DistillationDataset,
    PromptDistillationCollator,
    filter_records,
    load_jsonl_records,
)
from where_to_bake.eval import evaluate_model
from where_to_bake.models import create_teacher_student_pair
from where_to_bake.models.wrapper import _require_hf_stack, load_base_models, load_tokenizer
from where_to_bake.train.losses import compute_token_kl
from where_to_bake.utils.io import save_json, save_resolved_config, validate_result_schema
from where_to_bake.utils.metrics import estimate_adapter_bytes, get_model_trainable_params
from where_to_bake.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _build_dataset(
    tokenizer: Any,
    config: dict[str, Any],
    split_path: str,
    paraphrase_split: str | None = None,
    family_scope: str | None = None,
) -> DistillationDataset:
    records = load_jsonl_records(split_path)
    records = filter_records(
        records=records,
        family_scope=family_scope or config["baseline"].get("family_scope", "all"),
        paraphrase_split=paraphrase_split,
    )
    return DistillationDataset(
        tokenizer=tokenizer,
        records=records,
        max_source_length=config["data"]["max_source_length"],
        max_target_length=config["data"]["max_target_length"],
        prompting_config=config["prompting"],
    )


def _build_dataloader(dataset: DistillationDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PromptDistillationCollator(dataset.tokenizer),
    )


def _maybe_save_predictions(predictions: list[dict[str, Any]], output_dir: Path) -> str | None:
    if not predictions:
        return None
    path = output_dir / "predictions.json"
    save_json(path, predictions)
    return str(path)


def run_experiment(config: dict[str, Any], override_mode: str | None = None) -> dict[str, Any]:
    """Train and evaluate a registered baseline."""

    mode = override_mode or config["run"]["mode"]
    output_dir = Path(config["output"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["run"]["seed"])
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    torch, AutoModelForCausalLM, AutoTokenizer, _peft_items = _require_hf_stack()
    tokenizer = load_tokenizer(config, AutoTokenizer)

    train_dataset = _build_dataset(
        tokenizer,
        config,
        config["data"]["train_path"],
        paraphrase_split=config["train"].get("train_paraphrase_split", "seen"),
    )
    valid_dataset = _build_dataset(
        tokenizer,
        config,
        config["data"]["valid_path"],
        paraphrase_split=config["eval"].get("valid_paraphrase_split", "seen"),
    )
    test_dataset = _build_dataset(
        tokenizer,
        config,
        config["data"]["test_path"],
        paraphrase_split=config["eval"].get("paraphrase_split", "all"),
    )
    preserve_loader = None
    if config["data"].get("preserve_path"):
        preserve_dataset = _build_dataset(
            tokenizer,
            config,
            config["data"]["preserve_path"],
            paraphrase_split="all",
            family_scope="all",
        )
        preserve_loader = _build_dataloader(
            preserve_dataset,
            batch_size=config["train"]["per_device_eval_batch_size"],
        )

    teacher_probe_model, student_probe_base = load_base_models(config, AutoModelForCausalLM)
    teacher_probe_model.to(config["run"]["device"])
    student_probe_base.to(config["run"]["device"])
    selection_result = resolve_baseline_selection(
        config=config,
        teacher_model=teacher_probe_model,
        base_model=student_probe_base,
        train_dataset=train_dataset,
        device=torch.device(config["run"]["device"]),
        torch=torch,
    )
    del teacher_probe_model
    del student_probe_base
    if torch.cuda.is_available() and config["run"]["device"].startswith("cuda"):
        torch.cuda.empty_cache()

    teacher_model, student_model, tokenizer, torch = create_teacher_student_pair(
        config=config,
        device=config["run"]["device"],
        target_modules=selection_result.target_modules,
    )

    train_loader = _build_dataloader(
        train_dataset,
        batch_size=config["train"]["per_device_train_batch_size"],
    )
    valid_loader = _build_dataloader(
        valid_dataset,
        batch_size=config["train"]["per_device_eval_batch_size"],
    )
    test_loader = _build_dataloader(
        test_dataset,
        batch_size=config["train"]["per_device_eval_batch_size"],
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    train_runtime = 0.0
    train_tokens = 0
    total_steps = 0
    last_train_loss = None
    if mode in {"train", "train_eval"}:
        start_time = time.perf_counter()
        student_model.train()
        teacher_model.eval()
        optimizer.zero_grad()
        max_steps = config["train"]["max_steps"]
        grad_accum = max(config["train"]["gradient_accumulation_steps"], 1)
        for epoch in range(max(config["train"]["num_epochs"], 1)):
            for batch in train_loader:
                teacher_inputs = {
                    "input_ids": batch["teacher_input_ids"].to(config["run"]["device"]),
                    "attention_mask": batch["teacher_attention_mask"].to(config["run"]["device"]),
                }
                student_inputs = {
                    "input_ids": batch["student_input_ids"].to(config["run"]["device"]),
                    "attention_mask": batch["student_attention_mask"].to(config["run"]["device"]),
                }
                with torch.no_grad():
                    teacher_outputs = teacher_model(**teacher_inputs)
                student_outputs = student_model(**student_inputs)
                loss = compute_token_kl(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    batch["student_response_mask"].to(config["run"]["device"]),
                    batch["teacher_response_mask"].to(config["run"]["device"]),
                    temperature=config["loss"]["temperature"],
                )
                (loss / grad_accum).backward()
                if (total_steps + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        config["train"]["max_grad_norm"],
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                train_tokens += int(batch["student_response_mask"].sum().item())
                total_steps += 1
                last_train_loss = float(loss.item())
                if total_steps % config["train"]["log_every_n_steps"] == 0:
                    LOGGER.info("step=%s train_kl=%.4f", total_steps, last_train_loss)
                if total_steps >= max_steps:
                    break
            if total_steps >= max_steps:
                break
        train_runtime = time.perf_counter() - start_time

    eval_loader = valid_loader if mode == "train" else test_loader
    eval_summary = evaluate_model(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        preserve_loader=preserve_loader,
        device=torch.device(config["run"]["device"]),
        temperature=config["loss"]["temperature"],
        max_eval_batches=config["eval"]["max_eval_batches"],
        max_new_tokens=config["eval"]["max_new_tokens"],
        style_word_limit=config["eval"]["style_word_limit"],
    )
    prediction_path = None
    if config["output"].get("save_predictions"):
        prediction_path = _maybe_save_predictions(eval_summary["predictions"], output_dir)

    trainable_params = get_model_trainable_params(student_model)
    train_tokens_per_sec = train_tokens / max(train_runtime, 1e-6) if train_runtime else 0.0
    peak_memory_mb = 0.0
    if torch.cuda.is_available() and config["run"]["device"].startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    result = {
        "run_name": config["run"]["run_name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": "unknown",
        "baseline_name": config["baseline"]["name"],
        "model_name": config["model"]["base_model_name_or_path"],
        "seed": config["run"]["seed"],
        "prompt_family": config["baseline"].get("family_scope", "all"),
        "paraphrase_split": config["eval"].get("paraphrase_split", "all"),
        "trainable_params": trainable_params,
        "train_runtime_sec": train_runtime,
        "peak_memory_mb": peak_memory_mb,
        "teacher_fidelity_metrics": eval_summary["teacher_fidelity_metrics"],
        "preservation_metrics": eval_summary["preservation_metrics"],
        "efficiency_metrics": {
            "trainable_params": trainable_params,
            "estimated_adapter_bytes": estimate_adapter_bytes(trainable_params),
            "train_tokens_per_sec": train_tokens_per_sec,
            "eval_tokens_per_sec": eval_summary["eval_tokens_per_sec"],
            "inference_latency_ms": None,
        },
        "config_path": config["config_path"],
        "resolved_config_path": str(output_dir / "resolved_config.yaml"),
        "selected_modules": selection_result.target_modules,
        "selection_strategy": selection_result.selection_strategy,
        "selection_budget": selection_result.selection_budget,
        "loss_weights": {
            "kl_weight": config["loss"]["kl_weight"],
            "delta_weight": config["loss"]["delta_weight"],
            "preserve_weight": config["loss"]["preserve_weight"],
        },
        "dataset_summary": {
            "train_examples": len(train_dataset),
            "valid_examples": len(valid_dataset),
            "test_examples": len(test_dataset),
        },
        "eval_summary_path": prediction_path,
        "notes": {
            "mode": mode,
            "last_train_loss": last_train_loss,
            "implemented_baseline": config["baseline"]["name"],
            "environment": {
                "device": config["run"]["device"],
            },
            "selection_notes": selection_result.notes,
        },
    }
    validate_result_schema(result)
    save_json(output_dir / "result.json", result)
    return result
