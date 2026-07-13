#!/usr/bin/env bash
# Quick train + eval smoke (tiny rule-based data, one game).
set -e

export EUNBI_GAME="${EUNBI_GAME:-blotto}"
export RUN_ID="${RUN_ID:-eunbi_smoke}"
export EUNBI_N_GAMES="${EUNBI_N_GAMES:-20}"
export EUNBI_EVAL_N="${EUNBI_EVAL_N:-5}"
export EUNBI_BLOTTO_N_GAMES="${EUNBI_BLOTTO_N_GAMES:-5}"
export EUNBI_EPOCHS="${EUNBI_EPOCHS:-1}"

exec "$(dirname "$0")/run_game.sh" "$@"
