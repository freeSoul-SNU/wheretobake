#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 scripts/generate_longform_dataset.py

BASELINES=(
  "configs/baselines/promptbake_kl_longform.yaml"
  "configs/baselines/random_subset_kl_longform.yaml"
  "configs/baselines/all_layer_lora_kl_longform.yaml"
)

for config_path in "${BASELINES[@]}"; do
  echo "Running ${config_path}"
  PYTHONPATH=src python3 -m where_to_bake.run --config "${config_path}"
done

PYTHONPATH=src python3 scripts/summarize_results.py --root-dir outputs --output-prefix outputs/summary/longform_runs

