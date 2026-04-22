"""Teacher/student model loading and LoRA attachment."""

from __future__ import annotations

from typing import Any


def _require_hf_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and peft are required to run the HF LoRA baseline. "
            "Install requirements.txt before running training."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, (LoraConfig, get_peft_model)


def create_teacher_student_pair(config: dict[str, Any], device: str) -> tuple[Any, Any, Any, Any]:
    """Load teacher model, student model, tokenizer, and the runtime torch module."""

    torch, AutoModelForCausalLM, AutoTokenizer, peft_items = _require_hf_stack()
    LoraConfig, get_peft_model = peft_items

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(model_config["base_model_name_or_path"])
    student_base = AutoModelForCausalLM.from_pretrained(model_config["base_model_name_or_path"])

    if model_config.get("gradient_checkpointing"):
        teacher_model.gradient_checkpointing_enable()
        student_base.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        target_modules=config["lora"]["target_modules"],
        task_type=config["lora"]["task_type"],
    )
    student_model = get_peft_model(student_base, lora_config)

    teacher_model.to(device)
    student_model.to(device)
    return teacher_model, student_model, tokenizer, torch

