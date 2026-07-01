"""Delayed credit assignment for long-horizon games."""

from __future__ import annotations

from typing import Literal


def assign_credit(
    steps: list,
    episode_bonus: float,
    *,
    gamma: float = 0.95,
    method: Literal["monte_carlo", "uniform", "last_step"] = "monte_carlo",
) -> None:
    eligible = [s for s in steps if s.eligible]
    if not eligible:
        return

    if method == "last_step":
        eligible[-1].credit = sum(s.raw_reward for s in steps) + episode_bonus
        for s in eligible[:-1]:
            s.credit = s.raw_reward
        return

    if method == "uniform":
        total = sum(s.raw_reward for s in steps) + episode_bonus
        share = total / len(eligible)
        for s in eligible:
            s.credit = share
        return

    # Monte Carlo: backward discounted returns on eligible steps only.
    running = episode_bonus
    for s in reversed(steps):
        if not s.eligible:
            continue
        running = s.raw_reward + gamma * running
        s.credit = running
