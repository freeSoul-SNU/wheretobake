#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_PREFIX="${ROOT_DIR}/.conda/where_to_bake"
TMP_DIR="${ROOT_DIR}/.conda/tmp"

mkdir -p "${TMP_DIR}"
export TMPDIR="${TMP_DIR}"
export PIP_NO_CACHE_DIR=1

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

if [ -d "${ENV_PREFIX}" ]; then
  echo "[conda] Updating environment at ${ENV_PREFIX}"
  "${CONDA_EXE}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  echo "[conda] Creating environment at ${ENV_PREFIX}"
  "${CONDA_EXE}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

echo
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_PREFIX}"
