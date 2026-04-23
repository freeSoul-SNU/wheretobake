"""Import smoke tests."""

from __future__ import annotations

import unittest

from where_to_bake.config import load_config


class ImportSmokeTest(unittest.TestCase):
    def test_main_baseline_configs_load(self) -> None:
        paths = [
            "configs/baselines/promptbake_kl.yaml",
            "configs/baselines/full_target_lora_kl.yaml",
            "configs/baselines/all_layer_lora_kl.yaml",
            "configs/baselines/random_subset_kl.yaml",
            "configs/baselines/magnitude_topk.yaml",
            "configs/baselines/gradient_topk.yaml",
            "configs/baselines/promptbake_kl_longform.yaml",
            "configs/baselines/promptbake_kl_longform_gpu.yaml",
        ]
        for path in paths:
            config = load_config(path)
            self.assertEqual(config["model"]["base_model_name_or_path"], "sshleifer/tiny-gpt2")
            self.assertIn("baseline", config)


if __name__ == "__main__":
    unittest.main()
