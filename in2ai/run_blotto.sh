#!/usr/bin/env bash
set -e

SCRIPT_DIR="${IN2AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=in2ai/lib.sh
source "${SCRIPT_DIR}/lib.sh"
fp_setup
fp_model_args

GAME=blotto
N_EPISODES="${N_EPISODES:-64}"
N_WORKERS="${N_WORKERS:-4}"

python in2ai/rollout/generate.py \
  --game "${GAME}" \
  --run_id "${RUN_ID}" \
  "${FP_MODEL_ARGS[@]}" \
  --n_episodes "${N_EPISODES}" \
  --n_workers "${N_WORKERS}"

python in2ai/train/prepare_dataset.py \
  --rollout_file "${FP_RUN_DIR}/${GAME}_rollouts.jsonl" \
  --out "${FP_RUN_DIR}/${GAME}_grpo_dataset.jsonl"

python in2ai/train/train_grpo.py \
  --game "${GAME}" \
  --run_id "${RUN_ID}" \
  --dataset_file "${FP_RUN_DIR}/${GAME}_grpo_dataset.jsonl" \
  ${FP_ADAPTER:+--adapter_path "${FP_ADAPTER}"}

fp_write_run_info "${GAME}"
echo "In2AI Blotto RL complete: ${FP_RUN_DIR}"
