#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cat <<'EOF'
[M3] Proposed selective method is not fully implemented yet.

Current status:
- selective baseline registry entry exists
- preselected module list can be accepted
- delta loss / preserve loss / full ablation configs are not implemented end-to-end

Planned entrypoints for this stage should eventually cover:
- selected module config input
- selective LoRA placement
- KL + delta + preserve training
- ablation runs

Please implement M3 components first, then replace this guard script with a runnable pipeline.
EOF

exit 1

