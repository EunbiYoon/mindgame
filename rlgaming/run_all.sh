#!/usr/bin/env bash
# Run RLGaming pipeline for all four MindGames environments.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for game in mafia blotto ipd codenames; do
  RLG_GAME="${game}" bash "${SCRIPT_DIR}/run_game.sh"
done
