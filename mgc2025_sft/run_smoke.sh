#!/usr/bin/env bash
# Quick end-to-end smoke test: convert → LoRA → eval (one game, tiny sample).
set -e

export MGC_SCRIPT_DIR="${MGC_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export MGC_MAX_TRAJ="${MGC_MAX_TRAJ:-10}"
export MGC_EVAL_N="${MGC_EVAL_N:-5}"
export RUN_ID="${RUN_ID:-mgc_smoke}"
export MGC_SMOKE_GAME="${MGC_SMOKE_GAME:-blotto}"

# shellcheck source=mgc2025_sft/lib.sh
source "${MGC_SCRIPT_DIR}/lib.sh"

mgc_setup_root

mgc_new_run_id
maf_train_lora_extra_args

echo "=== MGC2025 smoke: ${MGC_SMOKE_GAME} (max_trajectories=${MGC_MAX_TRAJ}, eval_n=${MGC_EVAL_N}) ==="
mgc_convert_game "${MGC_SMOKE_GAME}"

train="${MGC_DATA_DIR}/${MGC_SMOKE_GAME}_train.jsonl"
test="${MGC_DATA_DIR}/${MGC_SMOKE_GAME}_test.jsonl"
if [[ ! -s "${train}" ]]; then
  echo "FAIL: empty train file ${train}" >&2
  exit 1
fi
echo "OK convert: train=$(wc -l < "${train}") test=$(wc -l < "${test}")"

if [[ "${MGC_SMOKE_CONVERT_ONLY:-0}" == "1" ]]; then
  echo "MGC2025 smoke convert-only complete: ${MGC_DATA_DIR}"
  exit 0
fi

echo "=== train LoRA → lora/runs/${RUN_ID} ==="
python lora/train_lora.py \
  --train_file "${train}" \
  --run_id "${RUN_ID}" \
  "${TRAIN_LORA_EXTRA[@]}" \
  --epochs 1

echo "=== eval ==="
python mgc2025_sft/evaluate.py \
  --game "${MGC_SMOKE_GAME}" \
  --run_id "${RUN_ID}" \
  --model_dir "lora/runs/${RUN_ID}" \
  --test_file "${test}" \
  --n "${MGC_EVAL_N}"

mgc_write_run_info "smoke:${MGC_SMOKE_GAME}" "${train}" "${test}"

echo
echo "MGC2025 smoke complete."
echo "  data:  ${MGC_DATA_DIR}"
echo "  lora:  lora/runs/${RUN_ID}"
echo "  eval:  eval/runs/${RUN_ID}/metrics_mgc_${MGC_SMOKE_GAME}.json"
