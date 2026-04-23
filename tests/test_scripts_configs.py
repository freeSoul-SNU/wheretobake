"""Config-oriented tests for newly added experiment utilities."""

from __future__ import annotations

import unittest

from where_to_bake.config import load_config


class ScriptConfigTest(unittest.TestCase):
    def test_longform_dataset_config_points_to_generated_dir(self) -> None:
        config = load_config("configs/baselines/promptbake_kl_longform.yaml")
        self.assertIn("longform_v1", config["data"]["train_path"])
        self.assertEqual(config["run"]["dtype"], "float32")

    def test_gpu_config_overrides_device(self) -> None:
        config = load_config("configs/baselines/promptbake_kl_longform_gpu.yaml")
        self.assertEqual(config["run"]["device"], "cuda")
        self.assertEqual(config["run"]["dtype"], "bfloat16")


if __name__ == "__main__":
    unittest.main()
