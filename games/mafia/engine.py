"""Simplified Secret Mafia match engine for RL rollouts and validation."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Literal

ROLES = ["Mafia", "Detective", "Doctor", "Citizen", "Citizen"]
PLAYERS = ["A", "B", "C", "D", "E"]
MAX_DAYS = 5


def assign_roles(rng: random.Random) -> dict[str, str]:
    roles = ROLES[:]
    rng.shuffle(roles)
    return dict(zip(PLAYERS, roles))


def mafia_player(roles: dict[str, str]) -> str:
    return next(p for p, r in roles.items() if r == "Mafia")


def check_winner(roles: dict[str, str], alive: list[str]) -> Literal["mafia", "town"] | None:
    alive_roles = [roles[p] for p in alive]
    mafia_alive = sum(1 for r in alive_roles if r == "Mafia")
    town_alive = len(alive) - mafia_alive
    if mafia_alive == 0:
        return "town"
    if mafia_alive >= town_alive:
        return "mafia"
    return None


def parse_vote(text: str) -> str | None:
    if not text:
        return None
    vote = str(text).strip().upper()
    if vote in PLAYERS:
        return vote
    return None


def parse_mafia_action(obj: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    if not isinstance(action, dict):
        return None
    speak = str(action.get("speak", "")).strip()
    vote = parse_vote(str(action.get("vote", "")))
    if vote is None:
        return None
    return speak, vote


def mafia_action_valid(
    obj: dict[str, Any],
    *,
    valid_targets: list[str],
    require_speak: bool = False,
) -> bool:
    parsed = parse_mafia_action(obj)
    if parsed is None:
        return False
    speak, vote = parsed
    if require_speak and not speak:
        return False
    return vote in valid_targets


def tally_votes(votes: dict[str, str], rng: random.Random) -> str | None:
    if not votes:
        return None
    counts = Counter(votes.values())
    top = counts.most_common()
    if len(top) == 1:
        return top[0][0]
    if top[0][1] > top[1][1]:
        return top[0][0]
    tied = [target for target, count in top if count == top[0][1]]
    return rng.choice(tied)


def vote_outcome(
    vote: str,
    eliminated: str | None,
    roles: dict[str, str],
    self_id: str,
) -> Literal["correct", "wrong", "neutral", "invalid"]:
    if eliminated is None:
        return "neutral"
    if vote != eliminated:
        return "neutral"
    if roles[eliminated] == "Mafia":
        return "correct" if roles[self_id] != "Mafia" else "wrong"
    return "wrong" if roles[self_id] != "Mafia" else "correct"


def night_kill(
    roles: dict[str, str],
    alive: list[str],
    rng: random.Random,
    *,
    doctor_protect: str | None = None,
) -> str | None:
    mafia = mafia_player(roles)
    if mafia not in alive:
        return None
    targets = [p for p in alive if p != mafia and roles[p] != "Mafia"]
    if not targets:
        return None
    # Mafia prefers high-value roles.
    priority = [p for p in targets if roles[p] in ("Detective", "Doctor")]
    pool = priority or targets
    victim = rng.choice(pool)
    if victim == doctor_protect:
        return None
    return victim


def opponent_vote(
    skill: str,
    player: str,
    roles: dict[str, str],
    alive: list[str],
    rng: random.Random,
) -> str:
    others = [p for p in alive if p != player]
    if not others:
        return player
    mafia = mafia_player(roles)

    if skill == "naive":
        return rng.choice(others)

    if roles[player] == "Mafia":
        non_mafia = [p for p in others if roles[p] != "Mafia"]
        return rng.choice(non_mafia or others)

    if skill == "strong" and mafia in others:
        if roles[player] == "Detective" and rng.random() < 0.85:
            return mafia
        if roles[player] == "Doctor" and rng.random() < 0.7:
            return mafia
        if rng.random() < 0.75:
            return mafia

    if mafia in others and rng.random() < 0.6:
        return mafia
    return rng.choice(others)


def opponent_speak(
    skill: str,
    player: str,
    roles: dict[str, str],
    target: str,
    rng: random.Random,
) -> str:
    if roles[player] == "Mafia":
        if skill == "strong":
            return f"I am not convinced by the case on {target}. Their story looks inconsistent."
        return f"Let's hear more before we rush a vote on {target}."
    if roles[player] == "Detective":
        return f"My read is that {target} is suspicious based on night information."
    if roles[player] == "Doctor":
        return f"We should protect useful claims and watch {target} closely."
    return f"I want to compare vote history before deciding on {target}."


def build_prompt_input(
    *,
    day: int,
    alive: list[str],
    dead: list[str],
    public_chat: list[dict[str, str]],
    vote_history: list[dict[str, Any]],
    claims: dict[str, str],
    self_id: str,
    self_role: str,
) -> dict[str, Any]:
    valid_targets = [p for p in alive if p != self_id]
    return {
        "game_state": {
            "day": day,
            "alive": alive,
            "dead": dead,
            "public_chat": public_chat,
            "vote_history": vote_history,
            "claims": claims,
            "valid_targets": valid_targets,
        },
        "private_info": {"self_id": self_id, "self_role": self_role},
        "task": "Infer likely mafia and choose the next speech and vote.",
    }


def format_prompt(inp: dict[str, Any]) -> str:
    import json

    return (
        "You are playing Secret Mafia. Use opponent modeling and theory of mind.\n"
        "Given the state, infer hidden roles and choose speech/vote.\n\n"
        f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
        "Respond as JSON with keys: belief, reasoning, action."
    )
