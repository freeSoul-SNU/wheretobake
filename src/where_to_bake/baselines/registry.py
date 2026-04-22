"""Baseline registry following docs/BASELINES.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata for a baseline entry."""

    name: str
    extension: bool
    implemented: bool
    description: str


BASELINE_REGISTRY = {
    "promptbake_kl": BaselineSpec(
        name="promptbake_kl",
        extension=False,
        implemented=True,
        description="Prompt Baking style KL-only LoRA baseline.",
    ),
    "full_target_lora_kl": BaselineSpec(
        name="full_target_lora_kl",
        extension=False,
        implemented=False,
        description="Full target-module LoRA with KL only.",
    ),
    "all_layer_lora_kl": BaselineSpec(
        name="all_layer_lora_kl",
        extension=False,
        implemented=False,
        description="All-layer LoRA with KL only.",
    ),
    "random_subset_kl": BaselineSpec(
        name="random_subset_kl",
        extension=False,
        implemented=False,
        description="Random module subset with KL only.",
    ),
    "magnitude_topk": BaselineSpec(
        name="magnitude_topk",
        extension=False,
        implemented=False,
        description="Magnitude-based top-k module selection.",
    ),
    "gradient_topk": BaselineSpec(
        name="gradient_topk",
        extension=False,
        implemented=False,
        description="Gradient-based top-k module selection.",
    ),
    "ours_selective": BaselineSpec(
        name="ours_selective",
        extension=False,
        implemented=False,
        description="Mechanism-guided selective LoRA with KL, delta, and preserve.",
    ),
    "genpi_lite": BaselineSpec(
        name="genpi_lite",
        extension=True,
        implemented=False,
        description="GenPI-style extension baseline.",
    ),
    "opcd_refine": BaselineSpec(
        name="opcd_refine",
        extension=True,
        implemented=False,
        description="OPCD-style extension baseline.",
    ),
}


def get_baseline(name: str) -> BaselineSpec:
    """Return a baseline spec by name."""

    if name not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline '{name}'. Available baselines: {available}")
    return BASELINE_REGISTRY[name]

