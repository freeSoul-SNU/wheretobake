"""Prompt baking datasets and collators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows into memory."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_teacher_prompt(prompt_text: str, input_text: str) -> str:
    """Build teacher prompt prefix."""

    return f"System: {prompt_text}\nUser: {input_text}\nAssistant:"


def format_student_prompt(input_text: str) -> str:
    """Build student prompt prefix."""

    return f"User: {input_text}\nAssistant:"


@dataclass
class EncodedPair:
    """Tokenized teacher/student pair for distillation."""

    teacher_input_ids: list[int]
    teacher_attention_mask: list[int]
    teacher_response_mask: list[int]
    student_input_ids: list[int]
    student_attention_mask: list[int]
    student_response_mask: list[int]
    input_text: str
    target_text: str
    prompt_text: str
    prompt_family: str


class DistillationDataset(Dataset):
    """Distillation dataset for teacher prompted and student unprompted training."""

    def __init__(
        self,
        tokenizer: Any,
        records: list[dict[str, Any]],
        max_source_length: int,
        max_target_length: int,
        prompting_config: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.records = records
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompting_config = prompting_config
        self.eos_text = tokenizer.eos_token or ""

    def __len__(self) -> int:
        return len(self.records)

    def _encode_example(self, prefix_text: str, target_text: str) -> tuple[list[int], list[int], list[int]]:
        prefix_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_length,
        )["input_ids"]
        target_piece = f" {target_text}{self.eos_text}"
        target_ids = self.tokenizer(
            target_piece,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length,
        )["input_ids"]
        input_ids = prefix_ids + target_ids
        attention_mask = [1] * len(input_ids)
        response_mask = [0] * len(prefix_ids) + [1] * len(target_ids)
        return input_ids, attention_mask, response_mask

    def __getitem__(self, index: int) -> EncodedPair:
        row = self.records[index]
        prompt_text = row.get(self.prompting_config["prompt_field"], "")
        input_text = row[self.prompting_config["input_field"]]
        target_text = row.get(self.prompting_config["target_field"], "")
        prompt_family = row.get("prompt_family", "all")

        teacher_prefix = format_teacher_prompt(prompt_text, input_text)
        student_prefix = format_student_prompt(input_text)

        teacher_input_ids, teacher_attention_mask, teacher_response_mask = self._encode_example(
            teacher_prefix,
            target_text,
        )
        student_input_ids, student_attention_mask, student_response_mask = self._encode_example(
            student_prefix,
            target_text,
        )
        return EncodedPair(
            teacher_input_ids=teacher_input_ids,
            teacher_attention_mask=teacher_attention_mask,
            teacher_response_mask=teacher_response_mask,
            student_input_ids=student_input_ids,
            student_attention_mask=student_attention_mask,
            student_response_mask=student_response_mask,
            input_text=input_text,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_family=prompt_family,
        )


class PromptDistillationCollator:
    """Pad paired teacher/student sequences for distillation."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def _pad_stack(self, rows: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(row) for row in rows)
        padded = [row + [pad_value] * (max_len - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, batch: list[EncodedPair]) -> dict[str, Any]:
        teacher_input_ids = self._pad_stack([row.teacher_input_ids for row in batch], self.pad_token_id)
        teacher_attention_mask = self._pad_stack([row.teacher_attention_mask for row in batch], 0)
        teacher_response_mask = self._pad_stack([row.teacher_response_mask for row in batch], 0)
        student_input_ids = self._pad_stack([row.student_input_ids for row in batch], self.pad_token_id)
        student_attention_mask = self._pad_stack([row.student_attention_mask for row in batch], 0)
        student_response_mask = self._pad_stack([row.student_response_mask for row in batch], 0)

        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_response_mask": teacher_response_mask,
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "student_response_mask": student_response_mask,
            "input_texts": [row.input_text for row in batch],
            "target_texts": [row.target_text for row in batch],
            "prompt_texts": [row.prompt_text for row in batch],
            "prompt_families": [row.prompt_family for row in batch],
        }

