#!/usr/bin/env bash
# Shared LoRA train + eval (eunbi module).

EUNBI_DIR="${EUNBI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=scripts/lib.sh
source "${EUNBI_DIR}/../scripts/lib.sh"

export EUNBI_EVAL_N="${EUNBI_EVAL_N:-100}"
export EUNBI_BLOTTO_N_GAMES="${EUNBI_BLOTTO_N_GAMES:-30}"
export EUNBI_EPOCHS="${EUNBI_EPOCHS:-1}"
export EUNBI_N_GAMES="${EUNBI_N_GAMES:-200}"
export EUNBI_SEED="${EUNBI_SEED:-42}"
export EUNBI_DATA_SOURCE="${EUNBI_DATA_SOURCE:-games}"

eunbi_setup_root() {
  maf_setup_root
}

eunbi_validate_game() {
  case "$1" in
    mafia|blotto|ipd|codenames) return 0 ;;
    *)
      echo "Unknown EUNBI_GAME=${1}. Use: mafia blotto ipd codenames" >&2
      exit 1
      ;;
  esac
}

eunbi_new_run_id() {
  local game="${EUNBI_GAME:-game}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_eunbi_${game}}"
  export RUN_ID
  export EUNBI_DATA_DIR="${EUNBI_DATA_DIR:-eunbi/data/runs/${RUN_ID}}"
  mkdir -p "${EUNBI_DATA_DIR}"
}

eunbi_ensure_data() {
  local game="$1"
  local train="${EUNBI_DATA_DIR}/${game}_train.jsonl"
  local test="${EUNBI_DATA_DIR}/${game}_test.jsonl"

  if [[ -s "${train}" ]]; then
    if [[ ! -s "${test}" ]] && [[ "${game}" != "blotto" ]]; then
      cp "${train}" "${test}"
    fi
    return
  fi

  if [[ "${EUNBI_DATA_SOURCE}" == "mgc" ]]; then
    echo "=== convert MGC2025 ${game} → ${EUNBI_DATA_DIR} ==="
    # shellcheck source=mgc2025_sft/lib.sh
    source "${EUNBI_DIR}/../mgc2025_sft/lib.sh"
    export MGC_DATA_DIR="${EUNBI_DATA_DIR}"
    export MGC_MAX_TRAJ="${EUNBI_MGC_MAX_TRAJ:-${MGC_MAX_TRAJ}}"
    mgc_setup_root
    mgc_convert_game "${game}"
    return
  fi

  echo "=== generate ${game} data → ${EUNBI_DATA_DIR} ==="
  python "games/${game}/generate_sft.py" \
    --out "${train}" \
    --n_games "${EUNBI_N_GAMES}" \
    --seed "${EUNBI_SEED}"

  if [[ "${game}" != "blotto" ]]; then
    cp "${train}" "${test}"
  fi
}

eunbi_resolve_train_file() {
  local game="$1"
  if [[ -n "${EUNBI_TRAIN_FILE:-}" ]]; then
    echo "${EUNBI_TRAIN_FILE}"
    return
  fi
  if [[ -n "${EUNBI_DATA_DIR:-}" ]]; then
    echo "${EUNBI_DATA_DIR}/${game}_train.jsonl"
    return
  fi
  local default="games/${game}/sft.jsonl"
  if [[ -f "${default}" ]]; then
    echo "${default}"
    return
  fi
  echo "eunbi/data/runs/${RUN_ID}/${game}_train.jsonl"
}

eunbi_resolve_test_file() {
  local game="$1"
  local train_file="$2"
  if [[ -n "${EUNBI_TEST_FILE:-}" ]]; then
    echo "${EUNBI_TEST_FILE}"
    return
  fi
  if [[ -n "${EUNBI_DATA_DIR:-}" ]]; then
    echo "${EUNBI_DATA_DIR}/${game}_test.jsonl"
    return
  fi
  if [[ "${train_file}" == *_train.jsonl ]]; then
    echo "${train_file/_train.jsonl/_test.jsonl}"
    return
  fi
  echo "games/${game}/sft.jsonl"
}

eunbi_run_eval() {
  local game="$1"
  local test_file="$2"
  local model_dir="eunbi/lora/runs/${RUN_ID}"

  case "${game}" in
    blotto)
      python eunbi/eval/evaluate_blotto.py \
        --run_id "${RUN_ID}" \
        --model_dir "${model_dir}" \
        --n_games "${EUNBI_BLOTTO_N_GAMES}"
      ;;
    mafia)
      python eunbi/eval/evaluate_mafia.py \
        --run_id "${RUN_ID}" \
        --model_dir "${model_dir}" \
        --test_file "${test_file}" \
        --n "${EUNBI_EVAL_N}"
      ;;
    ipd)
      python eunbi/eval/evaluate_ipd.py \
        --run_id "${RUN_ID}" \
        --model_dir "${model_dir}" \
        --test_file "${test_file}" \
        --n "${EUNBI_EVAL_N}"
      ;;
    codenames)
      python eunbi/eval/evaluate_codenames.py \
        --run_id "${RUN_ID}" \
        --model_dir "${model_dir}" \
        --test_file "${test_file}" \
        --n "${EUNBI_EVAL_N}"
      ;;
  esac
}
