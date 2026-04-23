#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cat <<'EOF'
[M2] Localization pipeline is not fully implemented yet.

Current status:
- hooks/selection scaffolding exists
- full localization cache, within/across-family stability, and causal scoring are not implemented

Planned entrypoints for this stage should eventually cover:
- human-defined prompt family loader
- module delta extraction
- within_family_consistency
- across_family_similarity
- stability_score and causal_score logging

Please implement M2 components first, then replace this guard script with a runnable pipeline.
EOF

exit 1

