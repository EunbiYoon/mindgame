#!/usr/bin/env bash
# Generate RLGaming SFT data only (no LoRA train / eval).
set -e

SCRIPT_DIR="${RLGAMING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=rlgaming/lib.sh
source "${SCRIPT_DIR}/lib.sh"
maf_nfs_safe_exec "$@"
rlg_setup_root
rlg_new_run_id

GAME="${RLG_GAME:?Set RLG_GAME to one of: mafia blotto ipd codenames}"
rlg_validate_game "${GAME}"

FILTER_ARGS=()
if [[ "${RLG_FILTER_WIN_ONLY}" == "1" ]]; then
  FILTER_ARGS+=(--filter_win_only)
fi

python rlgaming/data/generate_sft.py \
  --game "${GAME}" \
  --out_dir "${RUN_DIR}" \
  --n_rule_games "${RLG_N_RULE_GAMES}" \
  --mgc_max_trajectories "${RLG_MGC_MAX_TRAJ}" \
  --test_frac "${RLG_MGC_TEST_FRAC}" \
  --seed "${RLG_MGC_SEED}" \
  --cache_dir "${HF_DATASETS_CACHE}" \
  --rule_weight "${RLG_RULE_WEIGHT}" \
  --mgc_weight "${RLG_MGC_WEIGHT}" \
  "${FILTER_ARGS[@]}"

rlg_write_run_info "${RUN_DIR}/${GAME}_train.jsonl" "${RUN_DIR}/${GAME}_test.jsonl" "${GAME}"
echo "RLGaming ${GAME} data run complete: ${RUN_DIR}"
