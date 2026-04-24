#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[M2] Generating long-form dataset if needed"
PYTHONPATH=src python3 scripts/generate_longform_dataset.py

echo "[M2] Running prompt similarity inspection"
PYTHONPATH=src python3 scripts/run_prompt_similarity.py --config configs/baselines/prompt_similarity_longform.yaml

echo "[M2] Analyzing localization report"
PYTHONPATH=src python3 scripts/analyze_prompt_similarity.py --input outputs/localization/prompt_similarity_longform.json

cat <<'EOF'
[M2] Partial localization output generated.

What this stage currently gives you:
- prompt-induced module delta collection
- within-family similarity
- across-family similarity
- preview stability score
- causal score proxy from module-wise delta ablation

What is still missing:
- full localization cache
- automatic selective LoRA placement from localization output

Artifacts:
- outputs/localization/prompt_similarity_longform.json
- outputs/localization/prompt_similarity_longform.csv
- outputs/localization/prompt_similarity_longform.analysis.json
EOF
