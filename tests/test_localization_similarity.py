"""Unit tests for within-family vs across-family similarity aggregation."""

from __future__ import annotations

import unittest

import torch

from where_to_bake.localization.similarity import DeltaExample, compute_similarity_report


class LocalizationSimilarityTest(unittest.TestCase):
    def test_within_family_similarity_exceeds_across_for_aligned_deltas(self) -> None:
        examples = [
            DeltaExample(
                source_id="doc1",
                prompt_family="concise",
                prompt_id="c1",
                prompt_text="Answer briefly.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([1.0, 0.0])},
            ),
            DeltaExample(
                source_id="doc1",
                prompt_family="concise",
                prompt_id="c2",
                prompt_text="Be concise.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([0.9, 0.1])},
            ),
            DeltaExample(
                source_id="doc1",
                prompt_family="formal",
                prompt_id="f1",
                prompt_text="Use a formal tone.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([-1.0, 0.0])},
            ),
        ]
        report = compute_similarity_report(examples, ["block.0"], alpha=0.5)
        concise_stats = report["stability_summary"]["concise"]["block.0"]
        self.assertGreater(concise_stats["within_family_consistency"], 0.9)
        self.assertLess(concise_stats["across_family_similarity"], 0.0)
        self.assertGreater(concise_stats["stability_score"], concise_stats["within_family_consistency"])
        self.assertEqual(report["pooling_strategy"], "response_last_k_concat")
        self.assertEqual(report["causal_metric"], "response_region_kl")

    def test_causal_scores_are_aggregated_into_combined_score(self) -> None:
        examples = [
            DeltaExample(
                source_id="doc1",
                prompt_family="formal",
                prompt_id="f1",
                prompt_text="Use a formal tone.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([1.0, 0.0])},
                causal_effects={"block.0": 0.4},
            ),
            DeltaExample(
                source_id="doc1",
                prompt_family="formal",
                prompt_id="f2",
                prompt_text="Write formally.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([0.9, 0.1])},
                causal_effects={"block.0": 0.6},
            ),
            DeltaExample(
                source_id="doc1",
                prompt_family="concise",
                prompt_id="c1",
                prompt_text="Be concise.",
                split="train",
                paraphrase_split="seen",
                module_deltas={"block.0": torch.tensor([-1.0, 0.0])},
                causal_effects={"block.0": 0.2},
            ),
        ]

        report = compute_similarity_report(examples, ["block.0"], alpha=0.5)
        formal_stats = report["stability_summary"]["formal"]["block.0"]

        self.assertAlmostEqual(formal_stats["causal_score"], 0.5, places=6)
        self.assertIsNotNone(formal_stats["combined_score"])
        self.assertGreater(formal_stats["combined_score"], 0.0)
        self.assertIsNotNone(formal_stats["selection_score"])


if __name__ == "__main__":
    unittest.main()
