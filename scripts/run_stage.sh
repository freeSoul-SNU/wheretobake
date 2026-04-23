#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_stage.sh <m0|m1|m2|m3|m4>"
  exit 2
fi

case "$1" in
  m0) bash scripts/run_stage_m0.sh ;;
  m1) bash scripts/run_stage_m1.sh ;;
  m2) bash scripts/run_stage_m2.sh ;;
  m3) bash scripts/run_stage_m3.sh ;;
  m4) bash scripts/run_stage_m4.sh ;;
  *)
    echo "Unknown stage: $1"
    echo "Usage: bash scripts/run_stage.sh <m0|m1|m2|m3|m4>"
    exit 2
    ;;
esac
