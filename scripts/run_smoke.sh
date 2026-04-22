#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl.yaml

