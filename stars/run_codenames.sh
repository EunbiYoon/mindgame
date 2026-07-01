#!/usr/bin/env bash
export STARS_GAME=codenames
exec "$(dirname "$0")/run_game.sh" "$@"
