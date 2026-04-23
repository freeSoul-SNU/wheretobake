"""Selection logic smoke tests."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from where_to_bake.baselines.selection import list_lora_candidates


class Conv1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):  # pragma: no cover - test helper
        return x


class _TinyMockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.c_attn = nn.Linear(4, 4)
        self.block.c_proj = Conv1D()
        self.block.c_fc = nn.Linear(4, 4)
        self.block.other = nn.ReLU()


class BaselineSelectionTest(unittest.TestCase):
    def test_candidate_modules_exist_for_supported_suffixes(self) -> None:
        model = _TinyMockModel()
        candidates = list_lora_candidates(model, ["c_attn", "c_proj", "c_fc"])
        self.assertEqual(set(candidates.keys()), {"block.c_attn", "block.c_fc", "block.c_proj"})

    def test_conv1d_like_modules_are_selected_when_class_name_matches(self) -> None:
        model = _TinyMockModel()
        candidates = list_lora_candidates(model, ["c_proj"])
        self.assertEqual(list(candidates.keys()), ["block.c_proj"])


if __name__ == "__main__":
    unittest.main()
