#!/usr/bin/env bash
# RLGaming-style MindGames (MindGames 3rd place): multi-source SFT + LoRA.
# All run outputs: rlgaming/runs/<RUN_ID>/

RLGAMING_DIR="${RLGAMING_DIR:-$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)}"
# shellcheck source=scripts/lib.sh
source "${RLGAMING_DIR}/../scripts/lib.sh"

export RLG_GAME="${RLG_GAME:-mafia}"
export RLG_BASE_MODEL="${RLG_BASE_MODEL:-${MAF_BASE_MODEL:-Qwen/Qwen3-8B}}"
export RLG_LORA_4BIT="${RLG_LORA_4BIT:-${MAF_LORA_4BIT:-1}}"
export RLG_N_RULE_GAMES="${RLG_N_RULE_GAMES:-100}"
export RLG_MGC_MAX_TRAJ="${RLG_MGC_MAX_TRAJ:-500}"
export RLG_MGC_TEST_FRAC="${RLG_MGC_TEST_FRAC:-0.1}"
export RLG_MGC_SEED="${RLG_MGC_SEED:-42}"
export RLG_MGC_CACHE_DIR="${RLG_MGC_CACHE_DIR:-mgc2025_sft/data/cache}"
export RLG_EVAL_N="${RLG_EVAL_N:-100}"
export RLG_FILTER_WIN_ONLY="${RLG_FILTER_WIN_ONLY:-1}"
export RLG_RULE_WEIGHT="${RLG_RULE_WEIGHT:-1}"
export RLG_MGC_WEIGHT="${RLG_MGC_WEIGHT:-2}"

rlg_setup_root() {
  export MAF_HF_CACHE="${MAF_HF_CACHE:-${RLG_MGC_CACHE_DIR}/huggingface}"
  maf_setup_root
}

rlg_validate_game() {
  local game="$1"
  case "${game}" in
    mafia|blotto|ipd|codenames) return 0 ;;
    *)
      echo "Unknown RLG_GAME=${game}. Use: mafia blotto ipd codenames" >&2
      exit 1
      ;;
  esac
}

rlg_new_run_id() {
  local game="${RLG_GAME:-mafia}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_rlg_${game}}"
  export RUN_ID
  RUN_DIR="rlgaming/runs/${RUN_ID}"
  mkdir -p "${RUN_DIR}" "${RUN_DIR}/lora"
  export RUN_DIR
}

rlg_train_lora_extra_args() {
  TRAIN_LORA_EXTRA=(--base_model "${RLG_BASE_MODEL}")
  if [[ "${RLG_LORA_4BIT}" == "1" ]]; then
    TRAIN_LORA_EXTRA+=(--load_in_4bit)
  fi
}

rlg_write_run_info() {
  local train_file="$1"
  local test_file="$2"
  local game="${3:-${RLG_GAME:-mafia}}"
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
    "approach": "RLGaming (MindGames Efficient 3rd place)",
    "game": titles.get("${game}", "${game}"),
    "game_key": "${game}",
    "base_model": "${RLG_BASE_MODEL}",
    "fine_tuning": "LoRA SFT",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_dir": "${RUN_DIR}",
    "train_file": "${train_file}",
    "test_file": "${test_file}",
    "lora_dir": "${RUN_DIR}/lora",
    "data_sources": ["rule_based", "mgc2025_proprietary"] if "${game}" == "mafia" else ["mgc2025_proprietary"],
    "n_rule_games": int("${RLG_N_RULE_GAMES}") if "${game}" == "mafia" else 0,
    "mgc_max_trajectories": int("${RLG_MGC_MAX_TRAJ}"),
    "filter_win_only": "${RLG_FILTER_WIN_ONLY}" == "1",
}
with open("${RUN_DIR}/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}
