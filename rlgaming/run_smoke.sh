#!/usr/bin/env bash
# Full pipeline smoke: tiny data + LoRA + eval (needs GPU).
set -e
cd "$(dirname "$0")/.."
export RLG_GAME="${RLG_GAME:-mafia}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_rlg_smoke_${RLG_GAME}}"
export RLG_N_RULE_GAMES=10
export RLG_MGC_MAX_TRAJ=20
export RLG_FILTER_WIN_ONLY=0
export RLG_EVAL_N=5
exec bash "rlgaming/run_${RLG_GAME}.sh"
