#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baselines/promptbake_kl.yaml}"
PYTHONPATH=src python3 -m where_to_bake.run --config "${CONFIG_PATH}" --mode eval

