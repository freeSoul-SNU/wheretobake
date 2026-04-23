"""Baseline-specific LoRA target-module selection."""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader, Subset

from where_to_bake.data import PromptDistillationCollator
from where_to_bake.train.losses import compute_token_kl


SUPPORTED_LORA_MODULE_TYPES = {"Linear", "Conv1D"}


@dataclass(frozen=True)
class SelectionResult:
    """Resolved LoRA module selection for a baseline."""

    target_modules: list[str]
    selection_strategy: str
    selection_budget: int | None
    notes: dict[str, Any]


def list_lora_candidates(model: Any, suffixes: list[str]) -> OrderedDict[str, Any]:
    """Return LoRA-compatible modules whose names end with supported suffixes."""

    matched: OrderedDict[str, Any] = OrderedDict()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if module.__class__.__name__ not in SUPPORTED_LORA_MODULE_TYPES:
            continue
        if any(module_name.endswith(suffix) for suffix in suffixes):
            matched[module_name] = module
    return matched


def _build_probe_loader(dataset: Any, batch_size: int, limit: int) -> DataLoader | None:
    if len(dataset) == 0:
        return None
    probe_count = min(limit, len(dataset))
    subset = Subset(dataset, list(range(probe_count)))
    return DataLoader(
        subset,
        batch_size=min(batch_size, probe_count),
        shuffle=False,
        collate_fn=PromptDistillationCollator(dataset.tokenizer),
    )


def _register_activation_hooks(model: Any, module_names: list[str], store: dict[str, Any]) -> list[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for module_name in module_names:
        module = module_map[module_name]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, name: str = module_name) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            store[name] = tensor.detach()

        handles.append(module.register_forward_hook(hook))
    return handles


def _compute_magnitude_scores(
    teacher_model: Any,
    base_model: Any,
    probe_loader: DataLoader | None,
    candidate_names: list[str],
    device: Any,
    torch: Any,
) -> dict[str, float]:
    scores = {name: 0.0 for name in candidate_names}
    if probe_loader is None:
        return scores

    teacher_handles: list[Any] = []
    base_handles: list[Any] = []
    try:
        for batch in probe_loader:
            teacher_store: dict[str, Any] = {}
            base_store: dict[str, Any] = {}
            teacher_handles = _register_activation_hooks(teacher_model, candidate_names, teacher_store)
            base_handles = _register_activation_hooks(base_model, candidate_names, base_store)
            teacher_model(
                input_ids=batch["teacher_input_ids"].to(device),
                attention_mask=batch["teacher_attention_mask"].to(device),
            )
            base_model(
                input_ids=batch["student_input_ids"].to(device),
                attention_mask=batch["student_attention_mask"].to(device),
            )
            for name in candidate_names:
                teacher_tensor = teacher_store.get(name)
                base_tensor = base_store.get(name)
                if teacher_tensor is None or base_tensor is None:
                    continue
                teacher_summary = teacher_tensor.float().mean(dim=(0, 1))
                base_summary = base_tensor.float().mean(dim=(0, 1))
                scores[name] += (teacher_summary - base_summary).norm(p=2).item()
            for handle in teacher_handles + base_handles:
                handle.remove()
            teacher_handles = []
            base_handles = []
    finally:
        for handle in teacher_handles + base_handles:
            handle.remove()
    return scores


def _compute_gradient_scores(
    teacher_model: Any,
    base_model: Any,
    probe_loader: DataLoader | None,
    candidate_modules: OrderedDict[str, Any],
    device: Any,
    torch: Any,
    temperature: float,
) -> dict[str, float]:
    scores = {name: 0.0 for name in candidate_modules}
    if probe_loader is None:
        return scores

    teacher_model.eval()
    base_model.train()
    for parameter in base_model.parameters():
        parameter.grad = None

    for batch in probe_loader:
        teacher_inputs = {
            "input_ids": batch["teacher_input_ids"].to(device),
            "attention_mask": batch["teacher_attention_mask"].to(device),
        }
        student_inputs = {
            "input_ids": batch["student_input_ids"].to(device),
            "attention_mask": batch["student_attention_mask"].to(device),
        }
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
        base_outputs = base_model(**student_inputs)
        loss = compute_token_kl(
            base_outputs.logits,
            teacher_outputs.logits,
            batch["student_response_mask"].to(device),
            batch["teacher_response_mask"].to(device),
            temperature=temperature,
        )
        loss.backward()

        for name, module in candidate_modules.items():
            grad_norm = 0.0
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name, None)
                if parameter is not None and parameter.grad is not None:
                    grad_norm += parameter.grad.detach().float().norm(p=2).item()
            scores[name] += grad_norm
        for parameter in base_model.parameters():
            parameter.grad = None
    return scores


def resolve_baseline_selection(
    config: dict[str, Any],
    teacher_model: Any,
    base_model: Any,
    train_dataset: Any,
    device: Any,
    torch: Any,
) -> SelectionResult:
    """Resolve the LoRA target modules for the selected baseline."""

    baseline_name = config["baseline"]["name"]
    selection_config = config.get("selection", {})
    candidate_suffixes = selection_config.get(
        "candidate_module_suffixes",
        config["lora"]["target_modules"],
    )
    candidate_modules = list_lora_candidates(base_model, candidate_suffixes)
    candidate_names = list(candidate_modules.keys())
    if not candidate_names:
        raise ValueError(
            "No LoRA candidate modules were found. "
            f"Requested suffixes: {candidate_suffixes}"
        )

    if baseline_name == "promptbake_kl":
        target_modules = list(config["lora"]["target_modules"])
        return SelectionResult(
            target_modules=target_modules,
            selection_strategy="configured",
            selection_budget=len(target_modules),
            notes={"candidate_count": len(candidate_names)},
        )

    if baseline_name == "full_target_lora_kl":
        target_modules = list(selection_config.get("full_target_module_suffixes", candidate_suffixes))
        return SelectionResult(
            target_modules=target_modules,
            selection_strategy="full_target_suffixes",
            selection_budget=len(target_modules),
            notes={"candidate_count": len(candidate_names)},
        )

    if baseline_name == "all_layer_lora_kl":
        return SelectionResult(
            target_modules=candidate_names,
            selection_strategy="all_layer_exact",
            selection_budget=len(candidate_names),
            notes={"candidate_count": len(candidate_names)},
        )

    budget = min(selection_config.get("budget", len(config["lora"]["target_modules"])), len(candidate_names))
    seed = config["run"]["seed"]

    if baseline_name == "random_subset_kl":
        rng = random.Random(seed)
        target_modules = sorted(rng.sample(candidate_names, k=budget))
        return SelectionResult(
            target_modules=target_modules,
            selection_strategy="random_subset_exact",
            selection_budget=budget,
            notes={"candidate_count": len(candidate_names)},
        )

    probe_loader = _build_probe_loader(
        dataset=train_dataset,
        batch_size=config["train"]["per_device_train_batch_size"],
        limit=selection_config.get("probe_examples", len(train_dataset)),
    )

    if baseline_name == "magnitude_topk":
        scores = _compute_magnitude_scores(
            teacher_model=teacher_model,
            base_model=base_model,
            probe_loader=probe_loader,
            candidate_names=candidate_names,
            device=device,
            torch=torch,
        )
        target_modules = sorted(scores, key=scores.get, reverse=True)[:budget]
        return SelectionResult(
            target_modules=target_modules,
            selection_strategy="magnitude_topk_exact",
            selection_budget=budget,
            notes={"candidate_count": len(candidate_names), "scores": scores},
        )

    if baseline_name == "gradient_topk":
        scores = _compute_gradient_scores(
            teacher_model=teacher_model,
            base_model=base_model,
            probe_loader=probe_loader,
            candidate_modules=candidate_modules,
            device=device,
            torch=torch,
            temperature=config["loss"]["temperature"],
        )
        target_modules = sorted(scores, key=scores.get, reverse=True)[:budget]
        return SelectionResult(
            target_modules=target_modules,
            selection_strategy="gradient_topk_exact",
            selection_budget=budget,
            notes={"candidate_count": len(candidate_names), "scores": scores},
        )

    if baseline_name == "ours_selective":
        selected_modules = selection_config.get("selected_modules")
        if not selected_modules:
            raise NotImplementedError(
                "ours_selective is only partially supported right now. "
                "Provide selection.selected_modules explicitly until localization is implemented."
            )
        return SelectionResult(
            target_modules=list(selected_modules),
            selection_strategy="preselected_exact",
            selection_budget=len(selected_modules),
            notes={"candidate_count": len(candidate_names)},
        )

    raise NotImplementedError(f"Selection strategy for baseline '{baseline_name}' is not implemented.")
