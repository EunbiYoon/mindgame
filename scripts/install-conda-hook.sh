#!/usr/bin/env bash
# One-time setup: source .env whenever you `conda activate maf`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate your conda env first: conda activate maf" >&2
  exit 1
fi

HOOK="${CONDA_PREFIX}/etc/conda/activate.d/maf-project.sh"
mkdir -p "$(dirname "${HOOK}")"
cat > "${HOOK}" <<EOF
#!/bin/bash
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export MAF_ROOT="${ROOT}"
cd "${ROOT}" 2>/dev/null || true
EOF
chmod +x "${HOOK}"
echo "Installed conda hook: ${HOOK}"
echo "Run: conda deactivate && conda activate maf"
