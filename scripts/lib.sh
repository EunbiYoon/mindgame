#!/usr/bin/env bash
# Shared helpers for MindGames pipeline scripts.

export MAF_BASE_MODEL="${MAF_BASE_MODEL:-Qwen/Qwen3-8B}"
export MAF_LORA_4BIT="${MAF_LORA_4BIT:-1}"

maf_nfs_safe_exec() { :; }

maf_train_lora_extra_args() {
  TRAIN_LORA_EXTRA=(--base_model "${MAF_BASE_MODEL}")
  if [[ "${MAF_LORA_4BIT}" == "1" ]]; then
    TRAIN_LORA_EXTRA+=(--load_in_4bit)
  fi
}

maf_load_env() {
  local root="${MAF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  export MAF_ROOT="${root}"
  if [[ -f "${MAF_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${MAF_ROOT}/.env"
    set +a
  fi
}

maf_setup_hf_cache() {
  local cache_root="${MAF_HF_CACHE:-games/.cache/huggingface}"
  mkdir -p "${cache_root}/datasets" "${cache_root}/hub"
  export HF_HOME="${cache_root}"
  export HF_DATASETS_CACHE="${cache_root}/datasets"
  export HUGGINGFACE_HUB_CACHE="${cache_root}/hub"
}

maf_setup_root() {
  local lib_dir="${MAF_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  ROOT="${MAF_ROOT:-$(cd "${lib_dir}/.." && pwd)}"
  export MAF_ROOT="${ROOT}"
  cd "${ROOT}"
  maf_load_env
  maf_setup_hf_cache
}

maf_new_run_id() {
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
  export RUN_ID
}

maf_write_run_info() {
  local game="$1"
  local train_file="$2"
  local run_dir="$3"
  python - <<PY
import json
from datetime import datetime, timezone

info = {
    "run_id": "${RUN_ID}",
    "game": "${game}",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "train_file": "${train_file}",
    "run_dir": "${run_dir}",
}
with open("${run_dir}/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}
