"""Metric helpers."""

from __future__ import annotations

from typing import Any


def get_model_trainable_params(model: Any) -> int:
    """Count trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_adapter_bytes(trainable_params: int) -> int:
    """Estimate adapter size with float32 parameters."""

    return int(trainable_params * 4)


def compute_style_agreement(
    families: list[str],
    generated_texts: list[str],
    word_limit: int,
) -> list[float]:
    """Rule-based style agreement scores for smoke evaluation."""

    scores: list[float] = []
    for family, text in zip(families, generated_texts):
        lowered = text.lower()
        word_count = len([word for word in text.split() if word])
        if family == "concise":
            scores.append(1.0 if word_count <= word_limit else 0.0)
        elif family == "formal":
            informal_markers = ["can't", "won't", "hey", "awesome"]
            scores.append(1.0 if not any(marker in lowered for marker in informal_markers) else 0.0)
        elif family == "step_by_step":
            scores.append(1.0 if ("1." in text or "step" in lowered) else 0.0)
        elif family == "refusal_safe":
            safe_markers = ["cannot", "can't", "unable", "safe"]
            scores.append(1.0 if any(marker in lowered for marker in safe_markers) else 0.0)
        else:
            scores.append(0.0)
    return scores

