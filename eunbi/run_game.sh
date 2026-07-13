#!/usr/bin/env bash
# LoRA train + eval for one game.
#
# Usage:
#   bash eunbi/run_blotto.sh
#   EUNBI_DATA_SOURCE=mgc bash eunbi/run_blotto.sh
#   EUNBI_TRAIN_FILE=path/to/train.jsonl bash eunbi/run_blotto.sh

set -e

export EUNBI_DIR="${EUNBI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=eunbi/lib.sh
source "${EUNBI_DIR}/lib.sh"
maf_nfs_safe_exec "$@"
eunbi_setup_root
eunbi_new_run_id
maf_train_lora_extra_args

GAME="${EUNBI_GAME:?Set EUNBI_GAME to one of: mafia blotto ipd codenames}"
eunbi_validate_game "${GAME}"

if [[ -z "${EUNBI_TRAIN_FILE:-}" ]]; then
  eunbi_ensure_data "${GAME}"
fi

TRAIN_FILE="$(eunbi_resolve_train_file "${GAME}")"
TEST_FILE="$(eunbi_resolve_test_file "${GAME}" "${TRAIN_FILE}")"
if [[ ! -s "${TRAIN_FILE}" ]]; then
  echo "FAIL: missing or empty train file: ${TRAIN_FILE}" >&2
  exit 1
fi

echo "=== train LoRA → eunbi/lora/runs/${RUN_ID} ==="
python eunbi/lora/train_lora.py \
  --train_file "${TRAIN_FILE}" \
  --run_id "${RUN_ID}" \
  "${TRAIN_LORA_EXTRA[@]}" \
  --epochs "${EUNBI_EPOCHS}"

echo "=== eval → eunbi/eval/runs/${RUN_ID} ==="
eunbi_run_eval "${GAME}" "${TEST_FILE}"

echo "eunbi ${GAME} run complete:"
echo "  lora: eunbi/lora/runs/${RUN_ID}"
echo "  eval: eunbi/eval/runs/${RUN_ID}"
