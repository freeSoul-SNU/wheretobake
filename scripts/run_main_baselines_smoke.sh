#!/usr/bin/env bash
set -euo pipefail

BASELINES=(
  "configs/baselines/promptbake_kl.yaml"
  "configs/baselines/full_target_lora_kl.yaml"
  "configs/baselines/all_layer_lora_kl.yaml"
  "configs/baselines/random_subset_kl.yaml"
  "configs/baselines/magnitude_topk.yaml"
  "configs/baselines/gradient_topk.yaml"
)

for config_path in "${BASELINES[@]}"; do
  echo "Running ${config_path}"
  PYTHONPATH=src python3 -m where_to_bake.run --config "${config_path}"
done
