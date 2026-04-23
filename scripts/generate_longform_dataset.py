#!/usr/bin/env python3
"""Generate long-form prompt-baking datasets from a seed corpus."""

from __future__ import annotations

import argparse
import json

from where_to_bake.data.longform_generator import generate_longform_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate long-form prompt-family datasets.")
    parser.add_argument(
        "--source-corpus",
        default="data/source_corpus/longform_seed_v1.yaml",
        help="Path to the seed long-form corpus YAML.",
    )
    parser.add_argument(
        "--prompt-family-spec",
        default="data/prompt_families/prompt_family_longform_v1.yaml",
        help="Path to the prompt family YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/datasets/longform_v1",
        help="Output directory for generated JSONL files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = generate_longform_dataset(
        source_corpus_path=args.source_corpus,
        prompt_family_spec_path=args.prompt_family_spec,
        output_dir=args.output_dir,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

