"""Registry smoke tests."""

from __future__ import annotations

import unittest

from where_to_bake.baselines import BASELINE_REGISTRY, get_baseline


class BaselineRegistryTest(unittest.TestCase):
    def test_expected_baselines_exist(self) -> None:
        expected = {
            "promptbake_kl",
            "full_target_lora_kl",
            "all_layer_lora_kl",
            "random_subset_kl",
            "magnitude_topk",
            "gradient_topk",
            "ours_selective",
            "genpi_lite",
            "opcd_refine",
        }
        self.assertTrue(expected.issubset(BASELINE_REGISTRY.keys()))

    def test_promptbake_is_implemented(self) -> None:
        self.assertTrue(get_baseline("promptbake_kl").implemented)

    def test_three_main_baselines_are_implemented(self) -> None:
        for baseline_name in [
            "promptbake_kl",
            "all_layer_lora_kl",
            "random_subset_kl",
        ]:
            self.assertTrue(get_baseline(baseline_name).implemented)


if __name__ == "__main__":
    unittest.main()
