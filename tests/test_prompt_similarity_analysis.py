"""Tests for prompt similarity analysis summary logic."""

from __future__ import annotations

import unittest

from scripts.analyze_prompt_similarity import analyze_prompt_similarity


class PromptSimilarityAnalysisTest(unittest.TestCase):
    def test_analysis_detects_within_greater_than_across_signal(self) -> None:
        report = {
            "generated_at": "2026-04-23T00:00:00+0900",
            "alpha": 0.5,
            "example_count": 6,
            "candidate_modules": ["m1", "m2"],
            "stability_summary": {
                "concise": {
                    "m1": {
                        "within_family_consistency": 0.8,
                        "across_family_similarity": 0.2,
                        "stability_score": 0.7,
                        "causal_score": 0.5,
                        "combined_score": 0.35,
                        "selection_score": 1.2,
                        "within_count": 3,
                        "across_count": 3,
                        "causal_count": 3,
                    },
                    "m2": {
                        "within_family_consistency": 0.3,
                        "across_family_similarity": 0.4,
                        "stability_score": 0.1,
                        "causal_score": 0.2,
                        "combined_score": 0.02,
                        "selection_score": -0.4,
                        "within_count": 3,
                        "across_count": 3,
                        "causal_count": 3,
                    },
                }
            },
        }

        summary = analyze_prompt_similarity(report)

        self.assertTrue(summary["signal_summary"]["signal_detected"])
        self.assertIn("concise", summary["signal_summary"]["families_with_within_gt_across"])
        self.assertIn("concise", summary["signal_summary"]["families_with_positive_combined_score"])
        self.assertEqual(summary["family_analysis"]["concise"]["positive_gap_module_count"], 1)
        self.assertEqual(summary["family_analysis"]["concise"]["positive_combined_module_count"], 2)
        self.assertEqual(summary["family_analysis"]["concise"]["strongest_module"]["module_name"], "m1")
        self.assertEqual(summary["family_analysis"]["concise"]["mean_selection_score"], 0.4)
        self.assertEqual(summary["family_analysis"]["concise"]["max_selection_score"], 1.2)
        self.assertAlmostEqual(summary["family_analysis"]["concise"]["selection_score_std"], 0.8)
        self.assertIn("z-scored within each family", summary["notes"]["selection_score"])


if __name__ == "__main__":
    unittest.main()
