"""Tests for long-form dataset generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from where_to_bake.data.longform_generator import generate_longform_dataset
from where_to_bake.data.prompt_dataset import load_jsonl_records


class LongformGeneratorTest(unittest.TestCase):
    def test_generate_longform_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            counts = generate_longform_dataset(
                source_corpus_path="data/source_corpus/longform_seed_v1.yaml",
                prompt_family_spec_path="data/prompt_families/prompt_family_longform_v1.yaml",
                output_dir=temp_dir,
            )
            self.assertGreaterEqual(counts["train"], 30)
            self.assertGreaterEqual(counts["valid"], 6)
            self.assertGreaterEqual(counts["test"], 12)
            self.assertGreaterEqual(counts["preserve"], 4)

            train_records = load_jsonl_records(Path(temp_dir) / "train.jsonl")
            self.assertTrue(all(len(row["input_text"].split()) > 60 for row in train_records[:3]))


if __name__ == "__main__":
    unittest.main()

