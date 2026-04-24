"""Tests for cross-function dataset generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from where_to_bake.data.longform_generator import generate_longform_dataset
from where_to_bake.data.prompt_dataset import load_jsonl_records


class CrossFunctionGeneratorTest(unittest.TestCase):
    def test_generate_cross_function_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            counts = generate_longform_dataset(
                source_corpus_path="data/source_corpus/cross_function_seed_v1.yaml",
                prompt_family_spec_path="data/prompt_families/prompt_family_cross_function_v1.yaml",
                output_dir=temp_dir,
            )
            self.assertGreaterEqual(counts["train"], 60)
            self.assertGreaterEqual(counts["valid"], 16)
            self.assertGreaterEqual(counts["test"], 32)
            self.assertGreaterEqual(counts["preserve"], 4)

            train_records = load_jsonl_records(Path(temp_dir) / "train.jsonl")
            family_names = {row["prompt_family"] for row in train_records}
            self.assertEqual(
                family_names,
                {"summary", "json_extract", "topic_label", "action_items"},
            )


if __name__ == "__main__":
    unittest.main()
