#!/usr/bin/env python3
"""Summarize prompt similarity results into an easier-to-read report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from where_to_bake.utils.io import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze prompt similarity report JSON.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to prompt similarity JSON report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for analyzed summary JSON.",
    )
    return parser


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def analyze_prompt_similarity(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact family-level summary from the raw similarity report."""

    family_analysis: dict[str, Any] = {}

    for family, module_rows in report.get("stability_summary", {}).items():
        comparable_rows = [
            {"module_name": module_name, **stats}
            for module_name, stats in module_rows.items()
            if stats.get("within_family_consistency") is not None
        ]
        within_values = [
            float(row["within_family_consistency"])
            for row in comparable_rows
            if row.get("within_family_consistency") is not None
        ]
        across_values = [
            float(row["across_family_similarity"])
            for row in comparable_rows
            if row.get("across_family_similarity") is not None
        ]
        causal_values = [
            float(row["causal_score"])
            for row in comparable_rows
            if row.get("causal_score") is not None
        ]
        combined_values = [
            float(row["combined_score"])
            for row in comparable_rows
            if row.get("combined_score") is not None
        ]
        selection_values = [
            float(row["selection_score"])
            for row in comparable_rows
            if row.get("selection_score") is not None
        ]
        gap_values = [
            float(row["within_family_consistency"]) - float(row["across_family_similarity"] or 0.0)
            for row in comparable_rows
        ]
        positive_gap_modules = sum(1 for gap in gap_values if gap > 0.0)
        positive_combined_modules = sum(
            1
            for row in comparable_rows
            if row.get("combined_score") is not None and float(row["combined_score"]) > 0.0
        )
        strongest_module = None
        if comparable_rows:
            strongest_row = max(
                comparable_rows,
                key=lambda row: (
                    row.get("combined_score") if row.get("combined_score") is not None else float("-inf"),
                    row.get("selection_score") if row.get("selection_score") is not None else float("-inf"),
                    row.get("stability_score") if row.get("stability_score") is not None else float("-inf"),
                ),
            )
            strongest_module = {
                "module_name": strongest_row["module_name"],
                "within_family_consistency": _round_or_none(strongest_row.get("within_family_consistency")),
                "across_family_similarity": _round_or_none(strongest_row.get("across_family_similarity")),
                "stability_score": _round_or_none(strongest_row.get("stability_score")),
                "causal_score": _round_or_none(strongest_row.get("causal_score")),
                "combined_score": _round_or_none(strongest_row.get("combined_score")),
                "selection_score": _round_or_none(strongest_row.get("selection_score")),
                "within_count": strongest_row.get("within_count"),
                "across_count": strongest_row.get("across_count"),
                "causal_count": strongest_row.get("causal_count"),
            }

        family_analysis[family] = {
            "comparable_module_count": len(comparable_rows),
            "mean_within_family_consistency": _round_or_none(_mean(within_values)),
            "mean_across_family_similarity": _round_or_none(_mean(across_values)),
            "mean_causal_score": _round_or_none(_mean(causal_values)),
            "mean_within_minus_across": _round_or_none(_mean(gap_values)),
            "mean_combined_score": _round_or_none(_mean(combined_values)),
            "mean_selection_score": _round_or_none(_mean(selection_values)),
            "max_selection_score": _round_or_none(max(selection_values) if selection_values else None),
            "selection_score_std": _round_or_none(
                (
                    sum((value - (_mean(selection_values) or 0.0)) ** 2 for value in selection_values) / len(selection_values)
                )
                ** 0.5
                if selection_values
                else None
            ),
            "positive_gap_module_count": positive_gap_modules,
            "positive_combined_module_count": positive_combined_modules,
            "strongest_module": strongest_module,
        }

    interpretable_families = [
        family
        for family, row in family_analysis.items()
        if row["mean_within_family_consistency"] is not None and row["mean_across_family_similarity"] is not None
    ]
    signal_families = [
        family
        for family in interpretable_families
        if family_analysis[family]["mean_within_family_consistency"] > family_analysis[family]["mean_across_family_similarity"]
    ]
    combined_signal_families = [
        family
        for family in interpretable_families
        if family_analysis[family]["mean_combined_score"] is not None and family_analysis[family]["mean_combined_score"] > 0.0
    ]

    return {
        "generated_at": report.get("generated_at"),
        "source_report": report.get("source_report"),
        "alpha": report.get("alpha"),
        "example_count": report.get("example_count"),
        "candidate_module_count": len(report.get("candidate_modules", [])),
        "pooling_strategy": report.get("pooling_strategy"),
        "response_last_k": report.get("response_last_k"),
        "causal_metric": report.get("causal_metric"),
        "family_analysis": family_analysis,
        "notes": {
            "selection_score": (
                "selection_score is z-scored within each family, so mean_selection_score is expected "
                "to be close to 0 and should not be used for comparing families."
            )
        },
        "signal_summary": {
            "interpretable_family_count": len(interpretable_families),
            "families_with_within_gt_across": signal_families,
            "families_with_positive_combined_score": combined_signal_families,
            "signal_detected": bool(signal_families),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["source_report"] = str(input_path.resolve())
    analysis = analyze_prompt_similarity(report)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.analysis.json")
    save_json(output_path, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
