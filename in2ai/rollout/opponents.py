"""Curriculum-based opponent selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


def load_curriculum(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("stages", []))


def stage_for_episode(stages: list[dict[str, Any]], episode_idx: int) -> dict[str, Any]:
    chosen = stages[0]
    for stage in stages:
        if episode_idx >= stage.get("min_episodes", 0):
            chosen = stage
    return chosen


def pick_opponent(
    game: str,
    rng: random.Random,
    stages: list[dict[str, Any]],
    episode_idx: int,
) -> str:
    stage = stage_for_episode(stages, episode_idx)
    game_cfg = stage.get(game, {})

    if game == "blotto":
        strategies = game_cfg.get("strategies", ["balanced"])
        return rng.choice(strategies)

    if game == "ipd":
        strategies = game_cfg.get("strategies", ["tit_for_tat"])
        return rng.choice(strategies)

    if game == "codenames":
        return game_cfg.get("opponent_skill", "random")

    if game == "mafia":
        skills = game_cfg.get("skills", ["heuristic"])
        return rng.choice(skills)

    return "default"
