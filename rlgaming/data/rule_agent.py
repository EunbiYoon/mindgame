"""Rule-based Secret Mafia agents (game-theory baselines for SFT data)."""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

ROLES = ["Mafia", "Detective", "Doctor", "Citizen", "Citizen"]
PLAYERS = ["A", "B", "C", "D", "E"]

MAFIA_LINES = [
    "I am not convinced by the case on {target}. Their story looks inconsistent.",
    "{target} is pushing too hard — that feels like a deflection.",
    "Let's hear more before we rush a vote on {target}.",
]
CITIZEN_LINES = [
    "I want to compare vote history before deciding on {target}.",
    "{target} changed their story; that is suspicious.",
    "We should pressure {target} based on what was said today.",
]
DETECTIVE_LINES = [
    "My read is that {target} is suspicious based on night information.",
    "I checked {target}; we should pressure them today.",
]
DOCTOR_LINES = [
    "We should protect useful claims and watch {target} closely.",
    "{target} is drawing heat; mafia may be exploiting that.",
]


def assign_roles(rng: random.Random) -> dict[str, str]:
    roles = ROLES[:]
    rng.shuffle(roles)
    return dict(zip(PLAYERS, roles))


def _mafia_player(roles: dict[str, str]) -> str:
    return next(p for p, r in roles.items() if r == "Mafia")


def _vote_target_rule(player: str, roles: dict[str, str], alive: list[str], rng: random.Random) -> str:
    others = [p for p in alive if p != player]
    mafia = _mafia_player(roles)

    if roles[player] == "Mafia":
        non_mafia = [p for p in others if roles[p] != "Mafia"]
        return rng.choice(non_mafia or others)

    # Village baseline: vote known/suspected mafia when not mafia
    if roles[player] != "Mafia" and mafia in others:
        return mafia

    return rng.choice(others)


def _speak_line(role: str, target: str, rng: random.Random) -> str:
    if role == "Mafia":
        pool = MAFIA_LINES
    elif role == "Detective":
        pool = DETECTIVE_LINES
    elif role == "Doctor":
        pool = DOCTOR_LINES
    else:
        pool = CITIZEN_LINES
    return rng.choice(pool).format(target=target)


def _belief(roles: dict[str, str], self_id: str, alive: list[str]) -> dict[str, dict[str, float]]:
    mafia = _mafia_player(roles)
    out: dict[str, dict[str, float]] = {}
    for p in alive:
        if p == self_id:
            continue
        if roles[self_id] == "Mafia":
            prob = 0.15 if roles[p] != "Mafia" else 0.05
        else:
            prob = 0.85 if p == mafia else 0.2
        out[p] = {"mafia": round(prob, 2)}
    return out


def make_rule_example(
    roles: dict[str, str],
    self_id: str,
    rng: random.Random,
    *,
    day: int = 1,
    game_id: int | None = None,
) -> dict[str, Any]:
    alive = PLAYERS[:]
    mafia = _mafia_player(roles)

    chat = []
    accusations: Counter[str] = Counter()
    claims: dict[str, str] = {}
    for p in alive:
        target = _vote_target_rule(p, roles, alive, rng)
        accusations[target] += 1
        text = _speak_line(roles[p], target, rng)
        chat.append({"speaker": p, "text": text})
        if roles[p] == "Detective" and rng.random() < 0.7:
            claims[p] = "Detective"
        elif roles[p] == "Mafia" and rng.random() < 0.25:
            claims[p] = rng.choice(["Citizen", "Detective"])
        else:
            claims[p] = "Citizen"

    votes = {p: _vote_target_rule(p, roles, alive, rng) for p in alive}
    vote = votes[self_id]

    if roles[self_id] == "Mafia":
        reasoning = "As mafia, redirect suspicion while voting a non-mafia player."
        speak = _speak_line("Mafia", vote, rng)
    else:
        reasoning = (
            f"Village baseline: highest suspicion on {mafia} from accusations and claims."
        )
        speak = _speak_line(roles[self_id], vote, rng)

    inp = {
        "game_state": {
            "day": day,
            "alive": alive,
            "dead": [],
            "public_chat": chat,
            "vote_history": [{"day": max(1, day - 1), "votes": votes}],
            "claims": claims,
        },
        "private_info": {"self_id": self_id, "self_role": roles[self_id]},
        "task": "Infer likely mafia and choose the next speech and vote.",
    }
    out = {
        "belief": _belief(roles, self_id, alive),
        "reasoning": reasoning,
        "action": {"speak": speak, "vote": vote},
    }

    raw_prompt = (
        "You are playing Secret Mafia.\n"
        f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
        "Respond as JSON with keys: belief, reasoning, action."
    )
    return {
        "prompt": raw_prompt,
        "completion": json.dumps(out, ensure_ascii=False),
        "input": inp,
        "output": out,
        "meta": {
            "source": "rule_based",
            "strategy": "game_theory_baseline",
            "self_role": roles[self_id],
            "day": day,
            "game_id": game_id,
        },
    }


def generate_rule_examples(n_games: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for gi in range(n_games):
        roles = assign_roles(rng)
        day = rng.randint(1, 3)
        for self_id in PLAYERS:
            examples.append(make_rule_example(roles, self_id, rng, day=day, game_id=gi))
    return examples
