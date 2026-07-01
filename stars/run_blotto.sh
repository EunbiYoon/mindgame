#!/usr/bin/env bash
export STARS_GAME=blotto
exec "$(dirname "$0")/run_game.sh" "$@"
