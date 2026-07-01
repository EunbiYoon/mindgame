#!/usr/bin/env python3
"""Generate Colonel Blotto SFT data (MindGames-style opponent modeling)."""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from engine import (  # noqa: E402
    FIELDS,
    MAX_ROUNDS,
    UNITS,
    WIN_ROUNDS,
    best_response_alloc,
    choose_opponent_strategy,
    field_preference_belief,
    format_allocation,
    infer_opponent_style,
    opponent_allocate,
    resolve_round,
)


def play_game(rng: random.Random, strategy: str) -> list[dict]:
    examples = []
    self_round_wins = opp_round_wins = 0
    round_history: list[dict] = []
    opp_history: list[dict] = []

    for round_idx in range(1, MAX_ROUNDS + 1):
        if self_round_wins >= WIN_ROUNDS or opp_round_wins >= WIN_ROUNDS:
            break

        opp_alloc = opponent_allocate(strategy, rng, round_idx - 1, opp_history)
        gold_alloc = best_response_alloc(opp_history, rng)
        belief = {
            "opponent_field_preference": field_preference_belief(opp_history),
            "opponent_style": infer_opponent_style(strategy, opp_history),
        }
        top_field = max(FIELDS, key=lambda f: belief["opponent_field_preference"][f])
        reasoning = (
            "No opponent history yet; open with a balanced allocation that can adapt."
            if not opp_history
            else (
                f"Opponent has favored field {top_field} in past rounds. "
                "Counter by contesting their strength and winning cheaper fields."
            )
        )
        action = {"allocation": format_allocation(gold_alloc)}
        game_state = {
            "game": "Colonel Blotto",
            "self_id": "Commander Alpha",
            "opponent_id": "Commander Beta",
            "round": round_idx,
            "max_rounds": MAX_ROUNDS,
            "rounds_won": {"self": self_round_wins, "opponent": opp_round_wins},
            "units_per_round": UNITS,
            "fields": list(FIELDS),
            "round_history": list(round_history),
            "rules": {
                "format": "[Ax By Cz]",
                "units_constraint": f"x+y+z={UNITS}",
                "round_win": "win 2+ of 3 fields",
                "match_win": f"first to {WIN_ROUNDS} round wins",
                "communication": False,
            },
        }
        inp = {
            "game_state": game_state,
            "task": "Model the opponent from past allocations and choose this round's allocation.",
        }
        out = {"belief": belief, "reasoning": reasoning, "action": action}
        prompt = (
            "You are Commander Alpha in Colonel Blotto (MindGames).\n"
            "Opponent modeling through repeated strategic interaction.\n"
            "2 players, zero-sum, no communication, up to 9 rounds.\n"
            "Each round allocate exactly 20 units across fields A, B, C.\n"
            "Action format: [A5 B10 C5]. Win 2+ fields to win the round.\n\n"
            f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
            "Respond as JSON with keys: belief, reasoning, action.\n"
            "action.allocation must be [Ax By Cz] with nonnegative integers summing to 20."
        )
        examples.append(
            {
                "prompt": prompt,
                "completion": json.dumps(out, ensure_ascii=False),
                "input": inp,
                "output": out,
                "meta": {
                    "opponent_strategy": strategy,
                    "round": round_idx,
                    "opponent_allocation": opp_alloc,
                    "gold_allocation": gold_alloc,
                },
            }
        )

        outcome = resolve_round(gold_alloc, opp_alloc)
        if outcome == "self":
            self_round_wins += 1
        elif outcome == "opponent":
            opp_round_wins += 1

        round_history.append(
            {
                "round": round_idx,
                "self_allocation": format_allocation(gold_alloc),
                "opponent_allocation": format_allocation(opp_alloc),
                "round_winner": outcome,
            }
        )
        opp_history.append(opp_alloc)

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="games/blotto/sft.jsonl")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_examples = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n_games):
            strategy = choose_opponent_strategy(rng)
            for ex in play_game(rng, strategy):
                row = {k: ex[k] for k in ("prompt", "completion", "input", "output", "meta")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_examples += 1

    print(f"Wrote {n_examples} examples to {args.out}")


if __name__ == "__main__":
    main()
