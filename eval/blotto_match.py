"""Full Colonel Blotto match evaluation (MindGames rules)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util

_spec = importlib.util.spec_from_file_location("blotto_engine", ROOT / "games" / "blotto" / "engine.py")
_blotto = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_blotto)

MAX_ROUNDS = _blotto.MAX_ROUNDS
UNITS = _blotto.UNITS
WIN_ROUNDS = _blotto.WIN_ROUNDS
choose_opponent_strategy = _blotto.choose_opponent_strategy
format_allocation = _blotto.format_allocation
is_valid_allocation = _blotto.is_valid_allocation
opponent_allocate = _blotto.opponent_allocate
parse_allocation = _blotto.parse_allocation
resolve_round = _blotto.resolve_round


def build_blotto_prompt(
    round_idx: int,
    self_round_wins: int,
    opp_round_wins: int,
    round_history: list[dict],
) -> str:
    inp = {
        "game_state": {
            "game": "Colonel Blotto",
            "self_id": "Commander Alpha",
            "opponent_id": "Commander Beta",
            "round": round_idx,
            "max_rounds": MAX_ROUNDS,
            "rounds_won": {"self": self_round_wins, "opponent": opp_round_wins},
            "units_per_round": UNITS,
            "fields": ["A", "B", "C"],
            "round_history": list(round_history),
            "rules": {
                "format": "[Ax By Cz]",
                "units_constraint": f"x+y+z={UNITS}",
                "round_win": "win 2+ of 3 fields",
                "match_win": f"first to {WIN_ROUNDS} round wins",
                "communication": False,
            },
        },
        "task": "Model the opponent from past allocations and choose this round's allocation.",
    }
    return (
        "You are Commander Alpha in Colonel Blotto (MindGames).\n"
        "Opponent modeling through repeated strategic interaction.\n"
        "2 players, zero-sum, no communication, up to 9 rounds.\n"
        "Each round allocate exactly 20 units across fields A, B, C.\n"
        "Action format: [A5 B10 C5]. Win 2+ fields to win the round.\n\n"
        f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
        "Respond as JSON with keys: belief, reasoning, action.\n"
        "action.allocation must be [Ax By Cz] with nonnegative integers summing to 20."
    )


def parse_model_allocation(text: str) -> dict[str, int] | None:
    from eval.metrics import extract_json

    obj = extract_json(text)
    if isinstance(obj, dict):
        action = obj.get("action", {})
        if isinstance(action, dict) and action.get("allocation"):
            alloc = parse_allocation(str(action["allocation"]))
            if alloc is not None and is_valid_allocation(alloc):
                return alloc
    return parse_allocation(text)


@dataclass
class BlottoMatchStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    clean_games: int = 0
    caused_errors: int = 0
    invalid_format: int = 0
    invalid_units: int = 0
    rounds_played: int = 0
    round_wins: int = 0
    per_match: list[dict] = field(default_factory=list)


def play_match(generate_fn, rng: random.Random, strategy: str) -> dict:
    self_round_wins = 0
    opp_round_wins = 0
    round_history: list[dict] = []
    opp_history: list[dict] = []
    caused_error = False
    invalid_format = False
    invalid_units = False
    rounds = 0
    round_wins = 0

    for round_idx in range(1, MAX_ROUNDS + 1):
        if self_round_wins >= WIN_ROUNDS or opp_round_wins >= WIN_ROUNDS:
            break

        opp_alloc = opponent_allocate(strategy, rng, round_idx - 1, opp_history)
        prompt = build_blotto_prompt(round_idx, self_round_wins, opp_round_wins, round_history)
        raw = generate_fn(prompt)
        alloc = parse_model_allocation(raw)

        rounds += 1
        if alloc is None:
            caused_error = True
            if parse_allocation(raw) is not None:
                invalid_units = True
            else:
                invalid_format = True
            return {
                "outcome": "loss",
                "caused_error": caused_error,
                "invalid_format": invalid_format,
                "invalid_units": invalid_units,
                "rounds": rounds,
                "round_wins": round_wins,
                "strategy": strategy,
            }

        outcome = resolve_round(alloc, opp_alloc)
        if outcome == "self":
            self_round_wins += 1
            round_wins += 1
        elif outcome == "opponent":
            opp_round_wins += 1

        round_history.append(
            {
                "round": round_idx,
                "self_allocation": format_allocation(alloc),
                "opponent_allocation": format_allocation(opp_alloc),
                "round_winner": outcome,
            }
        )
        opp_history.append(opp_alloc)

    if self_round_wins > opp_round_wins:
        match_outcome = "win"
    elif opp_round_wins > self_round_wins:
        match_outcome = "loss"
    else:
        match_outcome = "tie"

    return {
        "outcome": match_outcome,
        "caused_error": caused_error,
        "invalid_format": invalid_format,
        "invalid_units": invalid_units,
        "rounds": rounds,
        "round_wins": round_wins,
        "self_round_wins": self_round_wins,
        "opp_round_wins": opp_round_wins,
        "strategy": strategy,
    }


def run_blotto_matches(generate_fn, n_games: int, seed: int = 42) -> BlottoMatchStats:
    rng = random.Random(seed)
    stats = BlottoMatchStats()

    for _ in range(n_games):
        strategy = choose_opponent_strategy(rng)
        result = play_match(generate_fn, rng, strategy)
        stats.games += 1
        stats.rounds_played += result["rounds"]
        stats.round_wins += result["round_wins"]
        if not result["caused_error"]:
            stats.clean_games += 1
        else:
            stats.caused_errors += 1
            if result["invalid_format"]:
                stats.invalid_format += 1
            if result["invalid_units"]:
                stats.invalid_units += 1

        if result["outcome"] == "win":
            stats.wins += 1
        elif result["outcome"] == "loss":
            stats.losses += 1
        else:
            stats.ties += 1
        stats.per_match.append(result)

    return stats
