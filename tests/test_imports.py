"""Import smoke tests."""

from __future__ import annotations

import unittest

from where_to_bake.config import load_config


class ImportSmokeTest(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = load_config("configs/baselines/promptbake_kl.yaml")
        self.assertEqual(config["baseline"]["name"], "promptbake_kl")
        self.assertEqual(config["model"]["base_model_name_or_path"], "sshleifer/tiny-gpt2")


if __name__ == "__main__":
    unittest.main()
