#!/usr/bin/env bash
# In2AI first-place RL pipeline helpers.

export IN2AI_DIR="${IN2AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="${IN2AI_DIR}"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/lib.sh
source "${ROOT}/scripts/lib.sh"

fp_setup() {
  export MAF_LIB_DIR="${ROOT}/scripts"
  export IN2AI_DIR="${SCRIPT_DIR}"
  maf_setup_root
  export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
  export FP_RUN_DIR="in2ai/runs/${RUN_ID}"
  mkdir -p "${FP_RUN_DIR}"
}

fp_model_args() {
  FP_MODEL="${FP_MODEL:-${MAF_BASE_MODEL:-Qwen/Qwen3-8B}}"
  FP_ADAPTER="${FP_ADAPTER:-}"
  if [[ -n "${FP_ADAPTER}" ]]; then
    FP_MODEL_ARGS=(--model_dir "${FP_ADAPTER}")
  else
    FP_MODEL_ARGS=(--model_dir "${FP_MODEL}")
  fi
}

fp_write_run_info() {
  local game="$1"
  python - <<PY
import json
from datetime import datetime, timezone

info = {
    "run_id": "${RUN_ID}",
    "game": "${game}",
    "method": "in2ai_rl",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "base_model": "${MAF_BASE_MODEL:-Qwen/Qwen3-8B}",
    "run_dir": "${FP_RUN_DIR}",
}
with open("${FP_RUN_DIR}/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}
