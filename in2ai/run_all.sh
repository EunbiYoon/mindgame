#!/usr/bin/env bash
set -e

SCRIPT_DIR="${IN2AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=in2ai/lib.sh
source "${SCRIPT_DIR}/lib.sh"
maf_nfs_safe_exec "$@"

fp_setup

bash "${SCRIPT_DIR}/run_blotto.sh"
bash "${SCRIPT_DIR}/run_ipd.sh"
bash "${SCRIPT_DIR}/run_codenames.sh"
bash "${SCRIPT_DIR}/run_mafia.sh"

echo "In2AI 4-game RL pipeline complete: ${FP_RUN_DIR}"
