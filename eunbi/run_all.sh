#!/usr/bin/env bash
# Train + eval all four games (requires train jsonl per game).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for game in mafia blotto ipd codenames; do
  EUNBI_GAME="${game}" bash "${SCRIPT_DIR}/run_game.sh"
done
