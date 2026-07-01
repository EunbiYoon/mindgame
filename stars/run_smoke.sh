#!/usr/bin/env bash
# Ollama smoke: 1 test step, 1 ReAct round. Real inference, not dry-run.
set -e
cd "$(dirname "$0")/.."
export STARS_GAME="${STARS_GAME:-mafia}"
export STARS_EVAL_N=1
export STARS_MAX_REACT=1
export STARS_MAX_RETRIES=1
exec bash "stars/run_${STARS_GAME}.sh"
