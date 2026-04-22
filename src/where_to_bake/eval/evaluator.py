"""Evaluation loop for smoke runs."""

from __future__ import annotations

import time
from typing import Any

import torch

from where_to_bake.data.prompt_dataset import format_student_prompt
from where_to_bake.train.losses import compute_token_kl, compute_token_metrics
from where_to_bake.utils.metrics import compute_style_agreement


@torch.no_grad()
def _generate_texts(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    outputs: list[str] = []
    for input_text in batch["input_texts"]:
        prompt = format_student_prompt(input_text)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        completion = generated[0, prompt_len:]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def evaluate_model(
    teacher_model: Any,
    student_model: Any,
    tokenizer: Any,
    eval_loader: Any,
    preserve_loader: Any | None,
    device: torch.device,
    temperature: float,
    max_eval_batches: int,
    max_new_tokens: int,
    style_word_limit: int,
) -> dict[str, Any]:
    """Run teacher fidelity and preservation evaluation."""

    teacher_model.eval()
    student_model.eval()

    fidelity_kl = 0.0
    fidelity_agreement = 0.0
    total_batches = 0
    eval_tokens = 0
    style_scores: list[float] = []
    predictions: list[dict[str, str]] = []
    eval_start = time.perf_counter()

    for batch_index, batch in enumerate(eval_loader):
        if batch_index >= max_eval_batches:
            break

        teacher_inputs = {
            "input_ids": batch["teacher_input_ids"].to(device),
            "attention_mask": batch["teacher_attention_mask"].to(device),
        }
        student_inputs = {
            "input_ids": batch["student_input_ids"].to(device),
            "attention_mask": batch["student_attention_mask"].to(device),
        }
        teacher_outputs = teacher_model(**teacher_inputs)
        student_outputs = student_model(**student_inputs)

        kl_value = compute_token_kl(
            student_outputs.logits,
            teacher_outputs.logits,
            batch["student_response_mask"].to(device),
            batch["teacher_response_mask"].to(device),
            temperature=temperature,
        )
        token_metrics = compute_token_metrics(
            student_outputs.logits,
            teacher_outputs.logits,
            batch["student_response_mask"].to(device),
            batch["teacher_response_mask"].to(device),
        )
        generated_texts = _generate_texts(
            student_model,
            tokenizer,
            batch,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        batch_style_score = compute_style_agreement(
            families=batch["prompt_families"],
            generated_texts=generated_texts,
            word_limit=style_word_limit,
        )
        style_scores.extend(batch_style_score)
        predictions.extend(
            {
                "input_text": input_text,
                "prediction": prediction,
                "target_text": target_text,
                "prompt_family": prompt_family,
            }
            for input_text, prediction, target_text, prompt_family in zip(
                batch["input_texts"],
                generated_texts,
                batch["target_texts"],
                batch["prompt_families"],
            )
        )

        fidelity_kl += kl_value.item()
        fidelity_agreement += token_metrics["next_token_agreement"]
        total_batches += 1
        eval_tokens += int(batch["student_response_mask"].sum().item())

    eval_runtime = max(time.perf_counter() - eval_start, 1e-6)
    preserve_metrics = {
        "base_drift_kl": None,
        "unrelated_input_drift": None,
        "non_target_family_drop": None,
        "general_eval_drop": None,
    }
    if preserve_loader is not None:
        preserve_kl = 0.0
        preserve_agreement = 0.0
        preserve_batches = 0
        for batch_index, batch in enumerate(preserve_loader):
            if batch_index >= max_eval_batches:
                break
            teacher_inputs = {
                "input_ids": batch["student_input_ids"].to(device),
                "attention_mask": batch["student_attention_mask"].to(device),
            }
            student_inputs = {
                "input_ids": batch["student_input_ids"].to(device),
                "attention_mask": batch["student_attention_mask"].to(device),
            }
            teacher_outputs = teacher_model(**teacher_inputs)
            student_outputs = student_model(**student_inputs)
            preserve_kl += compute_token_kl(
                student_outputs.logits,
                teacher_outputs.logits,
                batch["student_response_mask"].to(device),
                batch["student_response_mask"].to(device),
                temperature=temperature,
            ).item()
            preserve_agreement += compute_token_metrics(
                student_outputs.logits,
                teacher_outputs.logits,
                batch["student_response_mask"].to(device),
                batch["student_response_mask"].to(device),
            )["next_token_agreement"]
            preserve_batches += 1
        if preserve_batches:
            preserve_metrics["base_drift_kl"] = preserve_kl / preserve_batches
            preserve_metrics["unrelated_input_drift"] = 1.0 - (preserve_agreement / preserve_batches)

    batch_count = max(total_batches, 1)
    return {
        "teacher_fidelity_metrics": {
            "token_kl": fidelity_kl / batch_count,
            "next_token_agreement": fidelity_agreement / batch_count,
            "task_accuracy": None,
            "task_reward": None,
            "style_agreement": sum(style_scores) / max(len(style_scores), 1),
            "teacher_match_rate": fidelity_agreement / batch_count,
        },
        "preservation_metrics": preserve_metrics,
        "eval_tokens_per_sec": eval_tokens / eval_runtime,
        "predictions": predictions,
    }

