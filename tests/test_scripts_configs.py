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

    def test_prompt_similarity_config_enables_localization_only(self) -> None:
        config = load_config("configs/baselines/prompt_similarity_longform.yaml")
        self.assertEqual(config["run"]["mode"], "localization_only")
        self.assertEqual(
            config["localization"]["output_prefix"],
            "outputs/localization/prompt_similarity_longform",
        )
        self.assertIn("longform_v1", config["data"]["train_path"])
        self.assertTrue(config["localization"]["compute_causal"])
        self.assertEqual(config["localization"]["representation_pooling"], "response_last_k_concat")
        self.assertEqual(config["localization"]["causal_metric"], "response_region_kl")

    def test_distilgpt2_prompt_similarity_config_loads(self) -> None:
        config = load_config("configs/baselines/prompt_similarity_longform_distilgpt2.yaml")
        self.assertEqual(config["model"]["base_model_name_or_path"], "distilgpt2")
        self.assertEqual(config["run"]["mode"], "localization_only")
        self.assertTrue(config["localization"]["compute_causal"])
        self.assertEqual(config["localization"]["response_last_k"], 4)

    def test_cross_function_prompt_similarity_config_loads(self) -> None:
        config = load_config("configs/baselines/prompt_similarity_cross_function_distilgpt2.yaml")
        self.assertEqual(config["model"]["base_model_name_or_path"], "distilgpt2")
        self.assertIn("cross_function_v1", config["data"]["train_path"])
        self.assertEqual(
            config["localization"]["output_prefix"],
            "outputs/localization/prompt_similarity_cross_function_distilgpt2",
        )
        self.assertEqual(config["localization"]["representation_pooling"], "response_last_k_concat")


if __name__ == "__main__":
    unittest.main()
