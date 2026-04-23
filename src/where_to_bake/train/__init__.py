"""Training utilities."""

from __future__ import annotations

from typing import Any

__all__ = ["run_experiment"]


def run_experiment(*args: Any, **kwargs: Any) -> Any:
    """Lazy import wrapper to avoid package-level cycles."""

    from .trainer import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
