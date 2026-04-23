"""Optional integration smoke tests for multiple baselines."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from where_to_bake.config import load_config
from where_to_bake.train import run_experiment


class MultiBaselineRunTest(unittest.TestCase):
    def test_run_three_main_baselines(self) -> None:
        if os.environ.get("WHERE_TO_BAKE_RUN_HF_TESTS") != "1":
            self.skipTest("Set WHERE_TO_BAKE_RUN_HF_TESTS=1 to run HF integration smoke tests.")

        baseline_paths = [
            "configs/baselines/promptbake_kl.yaml",
            "configs/baselines/all_layer_lora_kl.yaml",
            "configs/baselines/random_subset_kl.yaml",
        ]
        temp_root = tempfile.mkdtemp(prefix="where_to_bake_test_")
        try:
            for index, path in enumerate(baseline_paths):
                config = load_config(path)
                config["train"]["max_steps"] = 1
                config["train"]["num_epochs"] = 1
                config["eval"]["max_eval_batches"] = 1
                config["output"]["save_predictions"] = False
                config["output"]["output_dir"] = os.path.join(temp_root, f"run_{index}")
                result = run_experiment(config)
                self.assertEqual(result["baseline_name"], config["baseline"]["name"])
                self.assertIsNotNone(result["selection_strategy"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
