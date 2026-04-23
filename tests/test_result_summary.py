"""Tests for result summary utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from where_to_bake.utils.result_summary import build_summary_rows, write_summary_files


class ResultSummaryTest(unittest.TestCase):
    def test_summary_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "example_run"
            run_dir.mkdir(parents=True)
            payload = {
                "run_name": "demo",
                "baseline_name": "promptbake_kl",
                "model_name": "demo-model",
                "seed": 42,
                "prompt_family": "all",
                "paraphrase_split": "all",
                "trainable_params": 100,
                "train_runtime_sec": 1.0,
                "peak_memory_mb": 0.0,
                "teacher_fidelity_metrics": {
                    "token_kl": 0.1,
                    "next_token_agreement": 0.9,
                    "style_agreement": 0.8,
                },
                "preservation_metrics": {
                    "base_drift_kl": 0.01,
                    "unrelated_input_drift": 0.02,
                },
                "efficiency_metrics": {
                    "train_tokens_per_sec": 12.0,
                    "eval_tokens_per_sec": 20.0,
                },
                "selection_strategy": "configured",
                "selection_budget": 1,
            }
            with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            rows = build_summary_rows(root)
            self.assertEqual(len(rows), 1)
            json_path, csv_path = write_summary_files(rows, root / "summary" / "results")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()

