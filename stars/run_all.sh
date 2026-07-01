#!/usr/bin/env bash
# Run STARS eval for all four MindGames environments.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for game in mafia blotto ipd codenames; do
  STARS_GAME="${game}" bash "${SCRIPT_DIR}/run_game.sh"
done
