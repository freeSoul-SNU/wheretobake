"""CLI entrypoint for smoke baseline runs."""

from __future__ import annotations

import argparse
import json
import logging

from where_to_bake.baselines import get_baseline
from where_to_bake.config import load_config
from where_to_bake.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Run Where to Bake experiments.")
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval", "localization_only"],
        default=None,
        help="Optional override for run.mode.",
    )
    return parser


def main() -> None:
    """Program entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    baseline = get_baseline(config["baseline"]["name"])
    if not baseline.implemented:
        raise NotImplementedError(
            f"Baseline '{baseline.name}' is registered but not implemented yet."
        )
    result = run_experiment(config, override_mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

