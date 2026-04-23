"""Dataset filtering helpers."""

from __future__ import annotations

from typing import Any


def filter_records(
    records: list[dict[str, Any]],
    family_scope: str,
    paraphrase_split: str | None = None,
) -> list[dict[str, Any]]:
    """Filter records by prompt family and paraphrase split."""

    filtered = records
    if family_scope != "all":
        filtered = [row for row in filtered if row.get("prompt_family", "all") == family_scope]
    if paraphrase_split and paraphrase_split != "all":
        filtered = [row for row in filtered if row.get("paraphrase_split", "all") == paraphrase_split]
    return filtered
