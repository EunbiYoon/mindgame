#!/usr/bin/env bash
# STARS-style MindGames agent (Qwen3-8B via Ollama, no fine-tuning).
# Run outputs: stars/runs/<RUN_ID>/

STARS_DIR="${STARS_DIR:-$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)}"
# shellcheck source=scripts/lib.sh
source "${STARS_DIR}/../scripts/lib.sh"

export STARS_OLLAMA_HOST="${STARS_OLLAMA_HOST:-http://127.0.0.1:11434}"
export STARS_OLLAMA_MODEL="${STARS_OLLAMA_MODEL:-qwen3:8b}"
export STARS_OLLAMA_TIMEOUT="${STARS_OLLAMA_TIMEOUT:-300}"
export STARS_GAME="${STARS_GAME:-mafia}"
export STARS_EVAL_N="${STARS_EVAL_N:-50}"
export STARS_MAX_REACT="${STARS_MAX_REACT:-3}"
export STARS_MAX_RETRIES="${STARS_MAX_RETRIES:-2}"
export STARS_DATA_SOURCE="${STARS_DATA_SOURCE:-mgc2025}"
export STARS_MGC_MAX_TRAJ="${STARS_MGC_MAX_TRAJ:-500}"
export STARS_MGC_TEST_FRAC="${STARS_MGC_TEST_FRAC:-0.1}"
export STARS_MGC_SEED="${STARS_MGC_SEED:-42}"
export STARS_MGC_CACHE_DIR="${STARS_MGC_CACHE_DIR:-mgc2025_sft/data/cache}"

stars_setup_root() {
  export MAF_HF_CACHE="${MAF_HF_CACHE:-${STARS_MGC_CACHE_DIR}/huggingface}"
  maf_setup_root
}

stars_validate_game() {
  local game="$1"
  case "${game}" in
    mafia|blotto|ipd|codenames) return 0 ;;
    *)
      echo "Unknown STARS_GAME=${game}. Use: mafia blotto ipd codenames" >&2
      exit 1
      ;;
  esac
}

stars_new_run_id() {
  local game="${STARS_GAME:-mafia}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_${game}}"
  export RUN_ID
  RUN_DIR="stars/runs/${RUN_ID}"
  mkdir -p "${RUN_DIR}"
  export RUN_DIR
}

stars_resolve_test_file() {
  local game="${1:-${STARS_GAME:-mafia}}"
  if [[ -n "${STARS_TEST_FILE:-}" && -f "${STARS_TEST_FILE}" ]]; then
    echo "${STARS_TEST_FILE}"
    return 0
  fi
  local latest
  latest="$(ls -1dt mgc2025_sft/data/runs/*/"${game}"_test.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "${latest}" && -f "${latest}" ]]; then
    echo "${latest}"
    return 0
  fi
  echo ""
}

stars_ensure_test_data() {
  local game="${1:-${STARS_GAME:-mafia}}"
  local test_file
  test_file="$(stars_resolve_test_file "${game}")"
  if [[ -n "${test_file}" ]]; then
    echo "${test_file}"
    return 0
  fi
  if [[ "${STARS_DATA_SOURCE}" == "sft" && "${game}" == "mafia" ]]; then
    python games/mafia/generate_sft.py \
      --out "${RUN_DIR}/mafia_sft.jsonl" \
      --n_games 50 \
      --seed "${STARS_MGC_SEED}"
    echo "${RUN_DIR}/mafia_sft.jsonl"
    return 0
  fi
  python mgc2025_sft/convert.py \
    --game "${game}" \
    --out_dir "${RUN_DIR}/data" \
    --max_trajectories "${STARS_MGC_MAX_TRAJ}" \
    --test_frac "${STARS_MGC_TEST_FRAC}" \
    --seed "${STARS_MGC_SEED}" \
    --cache_dir "${HF_DATASETS_CACHE}"
  echo "${RUN_DIR}/data/${game}_test.jsonl"
}

stars_write_run_info() {
  local test_file="$1"
  local game="${2:-${STARS_GAME:-mafia}}"
  python - <<PY
import json
from datetime import datetime, timezone

titles = {
    "mafia": "Secret Mafia",
    "blotto": "Colonel Blotto",
    "ipd": "Three-Player IPD",
    "codenames": "Codenames",
}
info = {
    "run_id": "${RUN_ID}",
    "approach": "STARS (MindGames Efficient 2nd place)",
    "game": titles.get("${game}", "${game}"),
    "game_key": "${game}",
    "model": "${STARS_OLLAMA_MODEL}",
    "inference": "ollama",
    "fine_tuning": False,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_dir": "${RUN_DIR}",
    "test_file": "${test_file}",
    "eval_n": int("${STARS_EVAL_N}"),
    "max_react_steps": int("${STARS_MAX_REACT}"),
}
with open("${RUN_DIR}/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}
