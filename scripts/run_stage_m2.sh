#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[M2] Generating long-form dataset if needed"
PYTHONPATH=src python3 scripts/generate_longform_dataset.py

echo "[M2] Running prompt similarity inspection"
PYTHONPATH=src python3 scripts/run_prompt_similarity.py --config configs/baselines/prompt_similarity_longform.yaml

cat <<'EOF'
[M2] Partial localization output generated.

What this stage currently gives you:
- prompt-induced module delta collection
- within-family similarity
- across-family similarity
- preview stability score

What is still missing:
- causal score
- full localization cache
- automatic selective LoRA placement from localization output

Artifacts:
- outputs/localization/prompt_similarity_longform.json
- outputs/localization/prompt_similarity_longform.csv
EOF
