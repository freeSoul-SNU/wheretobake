#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.conda/where_to_bake"

if [ ! -d "${ENV_PREFIX}" ]; then
  "${ROOT_DIR}/scripts/setup_conda_env.sh"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [ -x "${HOME}/anaconda3/bin/conda" ]; then
  CONDA_EXE="${HOME}/anaconda3/bin/conda"
elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
  CONDA_EXE="${HOME}/miniconda3/bin/conda"
elif [ -x "/opt/conda/bin/conda" ]; then
  CONDA_EXE="/opt/conda/bin/conda"
else
  echo "conda command not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${ROOT_DIR}"

echo "[M2-X] Generating cross-function dataset"
PYTHONPATH=src python3 scripts/generate_longform_dataset.py \
  --source-corpus data/source_corpus/cross_function_seed_v1.yaml \
  --prompt-family-spec data/prompt_families/prompt_family_cross_function_v1.yaml \
  --output-dir data/datasets/cross_function_v1

echo "[M2-X] Running cross-function similarity inspection"
PYTHONPATH=src python3 scripts/run_prompt_similarity.py \
  --config configs/baselines/prompt_similarity_cross_function_distilgpt2.yaml

echo "[M2-X] Analyzing cross-function localization report"
PYTHONPATH=src python3 scripts/analyze_prompt_similarity.py \
  --input outputs/localization/prompt_similarity_cross_function_distilgpt2.json
