#!/usr/bin/env python3
"""Summarize baseline result JSON files."""

from __future__ import annotations

import argparse
import json

from where_to_bake.utils.result_summary import build_summary_rows, write_summary_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Where to Bake result JSON files.")
    parser.add_argument("--root-dir", default="outputs", help="Root output directory to scan.")
    parser.add_argument(
        "--output-prefix",
        default="outputs/summary/results_summary",
        help="Output prefix for .json and .csv summary files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_summary_rows(args.root_dir)
    json_path, csv_path = write_summary_files(rows, args.output_prefix)
    print(json.dumps({"rows": len(rows), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

