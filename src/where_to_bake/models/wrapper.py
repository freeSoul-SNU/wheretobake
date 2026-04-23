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


def _resolve_torch_dtype(torch: Any, dtype_name: str | None) -> Any:
    """Resolve a config dtype string to a torch dtype."""

    mapping = {
        None: None,
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported run.dtype value: {dtype_name}")
    return mapping[dtype_name]


def _build_lora_config(config: dict[str, Any], target_modules: list[str], base_model: Any, lora_items: Any) -> Any:
    LoraConfig, _get_peft_model = lora_items
    module_map = dict(base_model.named_modules())
    fan_in_fan_out = any(
        module.__class__.__name__ == "Conv1D"
        for module_name, module in module_map.items()
        if any(module_name == target or module_name.endswith(target) for target in target_modules)
    )
    return LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        target_modules=target_modules,
        task_type=config["lora"]["task_type"],
        fan_in_fan_out=fan_in_fan_out,
    )


def load_tokenizer(config: dict[str, Any], auto_tokenizer: Any) -> Any:
    """Load tokenizer with safe defaults."""

    model_config = config["model"]
    tokenizer = auto_tokenizer.from_pretrained(
        model_config["tokenizer_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_models(config: dict[str, Any], auto_model: Any) -> tuple[Any, Any]:
    """Load teacher and unfused student base model."""

    torch, _auto_model_cls, _auto_tokenizer_cls, _peft_items = _require_hf_stack()
    model_config = config["model"]
    model_load_kwargs = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "use_safetensors": model_config.get("use_safetensors", True),
        "torch_dtype": _resolve_torch_dtype(torch, config.get("run", {}).get("dtype")),
    }
    teacher_model = auto_model.from_pretrained(
        model_config["base_model_name_or_path"],
        **model_load_kwargs,
    )
    student_base = auto_model.from_pretrained(
        model_config["base_model_name_or_path"],
        **model_load_kwargs,
    )
    return teacher_model, student_base


def create_teacher_student_pair(
    config: dict[str, Any],
    device: str,
    target_modules: list[str] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load teacher model, student model, tokenizer, and the runtime torch module."""

    torch, AutoModelForCausalLM, AutoTokenizer, peft_items = _require_hf_stack()
    _lora_config_cls, get_peft_model = peft_items

    tokenizer = load_tokenizer(config, AutoTokenizer)
    teacher_model, student_base = load_base_models(config, AutoModelForCausalLM)
    model_config = config["model"]

    if model_config.get("gradient_checkpointing"):
        teacher_model.gradient_checkpointing_enable()
        student_base.gradient_checkpointing_enable()

    resolved_targets = target_modules or list(config["lora"]["target_modules"])
    lora_config = _build_lora_config(config, resolved_targets, student_base, peft_items)
    student_model = get_peft_model(student_base, lora_config)

    teacher_model.to(device)
    student_model.to(device)
    return teacher_model, student_model, tokenizer, torch
