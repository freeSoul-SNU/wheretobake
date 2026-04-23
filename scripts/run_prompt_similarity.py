#!/usr/bin/env python3
"""Inspect within-family and across-family prompt similarity."""

from __future__ import annotations

import argparse
import json

from where_to_bake.config import load_config
from where_to_bake.localization.similarity import (
    collect_prompt_deltas,
    compute_similarity_report,
    save_similarity_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt-family similarity inspection.")
    parser.add_argument("--config", required=True, help="Path to localization-capable config YAML.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix override for .json/.csv report files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    examples, module_names = collect_prompt_deltas(config)
    alpha = config.get("localization", {}).get("alpha", 0.5)
    report = compute_similarity_report(examples, module_names, alpha=alpha)
    output_prefix = args.output_prefix or config.get("localization", {}).get(
        "output_prefix",
        "outputs/localization/prompt_similarity",
    )
    json_path, csv_path = save_similarity_report(report, output_prefix)
    print(
        json.dumps(
            {
                "example_count": report["example_count"],
                "candidate_module_count": len(report["candidate_modules"]),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

