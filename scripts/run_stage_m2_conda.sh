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

bash scripts/run_stage_m2.sh
