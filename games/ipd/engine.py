"""Three-player Iterated Prisoner's Dilemma (MindGames / TextArena)."""

from __future__ import annotations

import random
import re
from typing import Literal

PLAYERS = ("P0", "P1", "P2")
ACTIONS = ("cooperate", "defect")
MAX_ROUNDS = 10
# MindGames / sal ipd-stage payoffs: T=5, R=3, P=1, S=0
PAYOFF = {
    ("cooperate", "cooperate"): (3, 3),
    ("cooperate", "defect"): (0, 5),
    ("defect", "cooperate"): (5, 0),
    ("defect", "defect"): (1, 1),
}

CHOICE_RE = re.compile(r"\[?\s*(cooperate|defect)\s*\]?", re.IGNORECASE)
MESSAGE_LINES = {
    "cooperate": [
        "I plan to cooperate this round if others do the same.",
        "Let's keep mutual cooperation going.",
        "Trust has paid off so far; I will cooperate.",
    ],
    "defect": [
        "I may need to protect myself if others defect.",
        "Someone looked unreliable last round.",
        "I will not cooperate blindly again.",
    ],
}


def parse_choice(text: str) -> str | None:
    if not text:
        return None
    m = CHOICE_RE.search(str(text))
    if not m:
        return None
    return m.group(1).lower()


def pairwise_payoff(a: str, b: str) -> tuple[int, int]:
    return PAYOFF[(a, b)]


def round_scores(actions: dict[str, str]) -> dict[str, int]:
    scores = {p: 0 for p in PLAYERS}
    for i, p1 in enumerate(PLAYERS):
        for p2 in PLAYERS[i + 1 :]:
            u1, u2 = pairwise_payoff(actions[p1], actions[p2])
            scores[p1] += u1
            scores[p2] += u2
    return scores


def choose_opponent_strategy(rng: random.Random) -> str:
    return rng.choice(
        ["always_cooperate", "always_defect", "tit_for_tat", "grim_trigger", "random"]
    )


def opponent_message(strategy: str, player: str, last_actions: dict[str, str], rng: random.Random) -> str:
    intent = opponent_action(strategy, player, last_actions, rng)
    return rng.choice(MESSAGE_LINES[intent])


def opponent_action(
    strategy: str, player: str, last_actions: dict[str, str], rng: random.Random
) -> str:
    if strategy == "always_cooperate":
        return "cooperate"
    if strategy == "always_defect":
        return "defect"
    if strategy == "random":
        return rng.choice(ACTIONS)
    if strategy == "grim_trigger":
        if any(last_actions.get(p) == "defect" for p in PLAYERS if p != player):
            return "defect"
        return "cooperate"
    # tit_for_tat: mirror P0's last action, else cooperate
    if not last_actions:
        return "cooperate"
    return last_actions.get("P0", "cooperate")


def opponent_belief(
    strategies: dict[str, str], last_actions: dict[str, str], self_id: str
) -> dict[str, dict[str, float]]:
    belief: dict[str, dict[str, float]] = {}
    for p in PLAYERS:
        if p == self_id:
            continue
        if strategies[p] in ("always_cooperate", "tit_for_tat") and last_actions.get(p, "cooperate") != "defect":
            belief[p] = {"cooperate": 0.8, "defect": 0.2}
        elif strategies[p] == "always_defect":
            belief[p] = {"cooperate": 0.1, "defect": 0.9}
        elif last_actions.get(p) == "defect":
            belief[p] = {"cooperate": 0.25, "defect": 0.75}
        else:
            belief[p] = {"cooperate": 0.55, "defect": 0.45}
    return belief


def gold_choice(
    strategies: dict[str, str],
    last_actions: dict[str, str],
    self_id: str,
    rng: random.Random,
) -> str:
    others = [p for p in PLAYERS if p != self_id]
    predicted = [opponent_action(strategies[p], p, last_actions, rng) for p in others]
    if all(a == "cooperate" for a in predicted):
        return "cooperate"
    if any(strategies[p] == "always_defect" for p in others):
        return "defect"
    if any(a == "defect" for a in predicted):
        return "defect"
    if not last_actions:
        return "cooperate"
    if any(last_actions.get(p) == "defect" for p in others):
        return "defect"
    return "cooperate"


def gold_message(choice: str, rng: random.Random) -> str:
    return rng.choice(MESSAGE_LINES[choice])


def format_choice(choice: str) -> str:
    return f"[{choice}]"


def simulated_round_reward(self_choice: str, opponent_actions: dict[str, str], self_id: str) -> int:
    actions = dict(opponent_actions)
    actions[self_id] = self_choice
    return round_scores(actions)[self_id]
