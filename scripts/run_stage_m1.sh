#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[M1] Generating long-form dataset"
PYTHONPATH=src python3 scripts/generate_longform_dataset.py

echo "[M1] Running config/selection/generator tests"
PYTHONPATH=src python3 -m unittest \
  tests/test_imports.py \
  tests/test_baseline_registry.py \
  tests/test_baseline_selection.py \
  tests/test_longform_generator.py \
  tests/test_result_summary.py \
  tests/test_scripts_configs.py

echo "[M1] Running main baseline comparison set"
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl_longform.yaml
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/random_subset_kl_longform.yaml
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/all_layer_lora_kl_longform.yaml

echo "[M1] Summarizing outputs"
PYTHONPATH=src python3 scripts/summarize_results.py --root-dir outputs --output-prefix outputs/summary/m1_summary

echo "[M1] Summary files:"
echo "  outputs/summary/m1_summary.json"
echo "  outputs/summary/m1_summary.csv"

