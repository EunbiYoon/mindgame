#!/usr/bin/env python3
"""Backward-compatible entry point for STARS Mafia eval."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    if "--game" not in sys.argv:
        sys.argv.insert(1, "--game")
        sys.argv.insert(2, "mafia")
    from stars.evaluate_game import main

    main()
