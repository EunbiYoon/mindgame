#!/usr/bin/env bash
export RLG_GAME=mafia
exec "$(dirname "$0")/run_game.sh" "$@"
