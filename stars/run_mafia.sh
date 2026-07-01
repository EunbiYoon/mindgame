#!/usr/bin/env bash
export STARS_GAME=mafia
exec "$(dirname "$0")/run_game.sh" "$@"
