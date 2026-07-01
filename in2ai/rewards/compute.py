"""Per-environment reward computation (In2AI environment-specific scales)."""

from __future__ import annotations

from typing import Any


def blotto_step_reward(
    outcome: str,
    *,
    round_win: float = 1.0,
    round_loss: float = -1.0,
    invalid: float = -2.0,
) -> float:
    if outcome == "invalid":
        return invalid
    if outcome == "self":
        return round_win
    if outcome == "opponent":
        return round_loss
    return 0.0


def blotto_episode_bonus(
    match_outcome: str,
    *,
    match_win: float = 5.0,
    match_loss: float = -5.0,
) -> float:
    if match_outcome == "win":
        return match_win
    if match_outcome == "loss":
        return match_loss
    return 0.0


def ipd_step_reward(
    round_score: int,
    *,
    score_scale: float = 15.0,
    invalid: float = -2.0,
) -> float:
    if round_score < 0:
        return invalid
    return round_score / score_scale


def ipd_episode_bonus(
    self_score: int,
    best_opponent_score: int,
    *,
    win_bonus: float = 3.0,
) -> float:
    if self_score > best_opponent_score:
        return win_bonus
    if self_score < best_opponent_score:
        return -win_bonus
    return 0.0


def codenames_step_reward(
    event: str,
    *,
    team_word: float = 1.0,
    wrong_team: float = -1.0,
    neutral: float = -0.5,
    assassin: float = -5.0,
    invalid: float = -2.0,
) -> float:
    table = {
        "team_word": team_word,
        "wrong_team": wrong_team,
        "neutral": neutral,
        "assassin": assassin,
        "invalid": invalid,
        "pass": 0.0,
    }
    return table.get(event, 0.0)


def mafia_step_reward(
    outcome: str,
    *,
    correct_vote: float = 1.0,
    wrong_vote: float = -1.0,
    neutral_vote: float = 0.0,
    invalid: float = -2.0,
) -> float:
    table = {
        "correct": correct_vote,
        "wrong": wrong_vote,
        "neutral": neutral_vote,
        "invalid": invalid,
    }
    return table.get(outcome, neutral_vote)


def mafia_episode_bonus(
    match_outcome: str,
    *,
    match_win: float = 5.0,
    match_loss: float = -5.0,
) -> float:
    if match_outcome == "win":
        return match_win
    if match_outcome == "loss":
        return match_loss
    return 0.0


def game_reward_config(cfg: dict[str, Any], game: str) -> dict[str, Any]:
    return dict(cfg.get("games", {}).get(game, {}))
