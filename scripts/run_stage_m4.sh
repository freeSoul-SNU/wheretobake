#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[M4] Summarizing available result files under outputs/"
PYTHONPATH=src python3 scripts/summarize_results.py --root-dir outputs --output-prefix outputs/summary/m4_summary

cat <<'EOF'
[M4] Current automation status:
- result JSON aggregation is available
- paper-style main/extension table generation is not fully implemented yet

Generated summary files:
- outputs/summary/m4_summary.json
- outputs/summary/m4_summary.csv
EOF

