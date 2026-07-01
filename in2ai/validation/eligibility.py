"""Eligibility gating: only strategically meaningful decisions receive credit."""

from __future__ import annotations


def is_eligible(game: str, phase: str, valid_output: bool) -> bool:
    if game == "blotto":
        return phase == "allocation"

    if game == "ipd":
        # Communication is observational; only simultaneous decisions get reward.
        return phase == "decision"

    if game == "codenames":
        return phase in ("spymaster", "operative")

    if game == "mafia":
        return phase == "vote"

    return valid_output


def filter_eligible_steps(steps: list) -> list:
    return [s for s in steps if getattr(s, "eligible", False)]
