#!/usr/bin/env bash
set -e

export MGC_SCRIPT_DIR="${MGC_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=mgc2025_sft/lib.sh
source "${MGC_SCRIPT_DIR}/lib.sh"
maf_nfs_safe_exec "$@"
mgc_setup_root
mgc_new_run_id codenames
maf_train_lora_extra_args

mgc_convert_game codenames

python lora/train_lora.py \
  --train_file "${MGC_DATA_DIR}/codenames_train.jsonl" \
  --run_id "${RUN_ID}" \
  "${TRAIN_LORA_EXTRA[@]}" \
  --epochs 1

python mgc2025_sft/evaluate.py \
  --game codenames \
  --run_id "${RUN_ID}" \
  --model_dir "lora/runs/${RUN_ID}" \
  --test_file "${MGC_DATA_DIR}/codenames_test.jsonl" \
  --n "${MGC_EVAL_N}"

mgc_write_run_info "Codenames" "${MGC_DATA_DIR}/codenames_train.jsonl" "${MGC_DATA_DIR}/codenames_test.jsonl"

echo "MGC2025 Codenames run complete: ${MGC_DATA_DIR}"
