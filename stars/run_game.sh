#!/usr/bin/env bash
# STARS MindGames eval for one game (Ollama / Qwen3-8B, no fine-tuning).
#
# Usage:
#   STARS_GAME=blotto bash stars/run_game.sh
#   STARS_EVAL_N=10 STARS_DRY_RUN=1 bash stars/run_game.sh

set -e

export STARS_DIR="${STARS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=stars/lib.sh
source "${STARS_DIR}/lib.sh"
maf_nfs_safe_exec "$@"
stars_setup_root
stars_new_run_id

GAME="${STARS_GAME:?Set STARS_GAME to one of: mafia blotto ipd codenames}"
stars_validate_game "${GAME}"

TEST_FILE="$(stars_ensure_test_data "${GAME}")"
stars_write_run_info "${TEST_FILE}" "${GAME}"

DRY_ARGS=()
if [[ "${STARS_DRY_RUN:-0}" == "1" ]]; then
  DRY_ARGS=(--dry_run)
fi

python stars/evaluate_game.py \
  --game "${GAME}" \
  --test_file "${TEST_FILE}" \
  --run_dir "${RUN_DIR}" \
  --n "${STARS_EVAL_N}" \
  --ollama_host "${STARS_OLLAMA_HOST}" \
  --ollama_model "${STARS_OLLAMA_MODEL}" \
  --max_react "${STARS_MAX_REACT}" \
  --max_retries "${STARS_MAX_RETRIES}" \
  "${DRY_ARGS[@]}"

echo "STARS ${GAME} run complete: ${RUN_DIR}"
