#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[M0] Running minimal promptbake_kl smoke baseline"
echo "[M0] Config: configs/baselines/promptbake_kl.yaml"

PYTHONPATH=src python3 -m unittest tests/test_imports.py tests/test_baseline_registry.py
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl.yaml

echo "[M0] Result: outputs/promptbake_kl_smoke/result.json"

