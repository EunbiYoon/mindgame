#!/usr/bin/env bash
export RLG_GAME=blotto
exec "$(dirname "$0")/run_game.sh" "$@"
