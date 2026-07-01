"""Colonel Blotto rules aligned with MindGames / TextArena."""

from __future__ import annotations

import random
import re
from typing import Literal

FIELDS = ("A", "B", "C")
UNITS = 20
MAX_ROUNDS = 9
WIN_ROUNDS = 5
ALLOC_RE = re.compile(
    r"\[?\s*A\s*(\d+)\s+B\s*(\d+)\s+C\s*(\d+)\s*\]?",
    re.IGNORECASE,
)


def empty_alloc() -> dict[str, int]:
    return {f: 0 for f in FIELDS}


def format_allocation(alloc: dict[str, int]) -> str:
    return f"[A{alloc['A']} B{alloc['B']} C{alloc['C']}]"


def parse_allocation(text: str) -> dict[str, int] | None:
    if not text:
        return None
    m = ALLOC_RE.search(str(text))
    if not m:
        return None
    alloc = {"A": int(m.group(1)), "B": int(m.group(2)), "C": int(m.group(3))}
    if any(alloc[f] < 0 for f in FIELDS):
        return None
    return alloc


def is_valid_allocation(alloc: dict[str, int], *, exact: bool = True) -> bool:
    if any(alloc.get(f, -1) < 0 for f in FIELDS):
        return False
    total = sum(alloc[f] for f in FIELDS)
    return total == UNITS if exact else total <= UNITS


def resolve_round(self_alloc: dict[str, int], opp_alloc: dict[str, int]) -> Literal["self", "opponent", "tie"]:
    self_wins = sum(1 for f in FIELDS if self_alloc[f] > opp_alloc[f])
    opp_wins = sum(1 for f in FIELDS if opp_alloc[f] > self_alloc[f])
    if self_wins >= 2:
        return "self"
    if opp_wins >= 2:
        return "opponent"
    return "tie"


def opponent_field_means(history: list[dict[str, int]]) -> dict[str, float]:
    if not history:
        return {f: UNITS / len(FIELDS) for f in FIELDS}
    return {f: sum(r[f] for r in history) / len(history) for f in FIELDS}


def field_preference_belief(history: list[dict[str, int]]) -> dict[str, float]:
    means = opponent_field_means(history)
    total = sum(means.values()) or 1.0
    return {f: round(means[f] / total, 2) for f in FIELDS}


def strategy_uniform(rng: random.Random) -> dict[str, int]:
    a = rng.randint(0, UNITS)
    b = rng.randint(0, UNITS - a)
    return {"A": a, "B": b, "C": UNITS - a - b}


def strategy_balanced(rng: random.Random) -> dict[str, int]:
    base = [7, 7, 6]
    rng.shuffle(base)
    return {"A": base[0], "B": base[1], "C": base[2]}


def strategy_field_focus(rng: random.Random, field: str | None = None) -> dict[str, int]:
    focus = field or rng.choice(FIELDS)
    alloc = empty_alloc()
    alloc[focus] = rng.randint(10, 14)
    rest = UNITS - alloc[focus]
    others = [f for f in FIELDS if f != focus]
    alloc[others[0]] = rng.randint(0, rest)
    alloc[others[1]] = rest - alloc[others[0]]
    return alloc


def strategy_cyclic(rng: random.Random, round_idx: int) -> dict[str, int]:
    return strategy_field_focus(rng, FIELDS[round_idx % len(FIELDS)])


def strategy_reactive(history: list[dict[str, int]], rng: random.Random) -> dict[str, int]:
    if not history:
        return strategy_balanced(rng)
    means = opponent_field_means(history)
    focus = max(FIELDS, key=lambda f: means[f])
    return strategy_field_focus(rng, focus)


def choose_opponent_strategy(rng: random.Random) -> str:
    return rng.choice(["uniform", "balanced", "field_focus", "cyclic", "reactive"])


def opponent_allocate(strategy: str, rng: random.Random, round_idx: int, history: list[dict[str, int]]) -> dict[str, int]:
    if strategy == "uniform":
        return strategy_uniform(rng)
    if strategy == "balanced":
        return strategy_balanced(rng)
    if strategy == "field_focus":
        return strategy_field_focus(rng)
    if strategy == "cyclic":
        return strategy_cyclic(rng, round_idx)
    if strategy == "reactive":
        return strategy_reactive(history, rng)
    return strategy_balanced(rng)


def best_response_alloc(opp_history: list[dict[str, int]], rng: random.Random) -> dict[str, int]:
    if not opp_history:
        return strategy_balanced(rng)
    means = opponent_field_means(opp_history)
    weak = min(FIELDS, key=lambda f: means[f])
    strong = max(FIELDS, key=lambda f: means[f])
    mid = [f for f in FIELDS if f not in (weak, strong)][0]
    alloc = empty_alloc()
    alloc[weak] = min(9, max(6, int(round(UNITS - means[weak]))))
    alloc[strong] = min(10, max(5, int(round(means[strong] + 1))))
    alloc[mid] = UNITS - alloc[weak] - alloc[strong]
    if alloc[mid] < 0:
        overflow = -alloc[mid]
        alloc[mid] = 0
        alloc[strong] = max(0, alloc[strong] - overflow)
        alloc[weak] = UNITS - alloc[strong] - alloc[mid]
    if not is_valid_allocation(alloc):
        return strategy_balanced(rng)
    return alloc


def infer_opponent_style(strategy: str, history: list[dict[str, int]]) -> str:
    if not history:
        return f"scripted:{strategy}"
    means = opponent_field_means(history)
    focus = max(FIELDS, key=lambda f: means[f])
    spread = max(means.values()) - min(means.values())
    if spread < 2.5:
        return "balanced_spender"
    return f"field_focus_{focus}"
