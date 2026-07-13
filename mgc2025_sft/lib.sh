#!/usr/bin/env bash
# MGC2025 trajectory pipeline (paper-style data, local LoRA + eval).

MGC_SCRIPT_DIR="${MGC_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=scripts/lib.sh
source "${MGC_SCRIPT_DIR}/../scripts/lib.sh"

# 0 = full dataset (slow download); dev default keeps runs tractable.
export MGC_MAX_TRAJ="${MGC_MAX_TRAJ:-500}"
export MGC_TEST_FRAC="${MGC_TEST_FRAC:-0.1}"
export MGC_SEED="${MGC_SEED:-42}"
export MGC_EVAL_N="${MGC_EVAL_N:-100}"
export MGC_CACHE_DIR="${MGC_CACHE_DIR:-mgc2025_sft/data/cache}"

mgc_setup_root() {
  export MAF_HF_CACHE="${MAF_HF_CACHE:-${MGC_CACHE_DIR}/huggingface}"
  maf_setup_root
}

mgc_new_run_id() {
  maf_new_run_id
  MGC_DATA_DIR="mgc2025_sft/data/runs/${RUN_ID}"
  mkdir -p "${MGC_DATA_DIR}" "${MGC_CACHE_DIR}"
}

mgc_convert_game() {
  local game="$1"
  python mgc2025_sft/convert.py \
    --game "${game}" \
    --out_dir "${MGC_DATA_DIR}" \
    --max_trajectories "${MGC_MAX_TRAJ}" \
    --test_frac "${MGC_TEST_FRAC}" \
    --seed "${MGC_SEED}" \
    --cache_dir "${HF_DATASETS_CACHE}"
}

mgc_convert_all() {
  python mgc2025_sft/convert.py \
    --game all \
    --out_dir "${MGC_DATA_DIR}" \
    --max_trajectories "${MGC_MAX_TRAJ}" \
    --test_frac "${MGC_TEST_FRAC}" \
    --seed "${MGC_SEED}" \
    --cache_dir "${HF_DATASETS_CACHE}"
}

mgc_write_run_info() {
  local game="$1"
  local train_file="$2"
  local test_file="$3"
  python - <<PY
import json
from datetime import datetime, timezone

info = {
    "run_id": "${RUN_ID}",
    "game": "${game}",
    "data_source": "mgc2025",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "train_file": "${train_file}",
    "test_file": "${test_file}",
    "mgc_data_dir": "${MGC_DATA_DIR}",
    "max_trajectories": int("${MGC_MAX_TRAJ}"),
    "lora_dir": "eunbi/lora/runs/${RUN_ID}",
    "eval_dir": "eunbi/eval/runs/${RUN_ID}",
}
with open("${MGC_DATA_DIR}/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}
