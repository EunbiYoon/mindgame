#!/usr/bin/env python3
"""Generate three-player IPD SFT data (MindGames)."""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from engine import (  # noqa: E402
    MAX_ROUNDS,
    PLAYERS,
    choose_opponent_strategy,
    format_choice,
    gold_choice,
    gold_message,
    opponent_action,
    opponent_belief,
    opponent_message,
    round_scores,
)


def play_game(rng: random.Random, strategies: dict[str, str], self_id: str) -> list[dict]:
    examples = []
    total_scores = {p: 0 for p in PLAYERS}
    last_actions: dict[str, str] = {}
    round_logs: list[dict] = []

    for round_idx in range(1, MAX_ROUNDS + 1):
        messages = {
            p: (
                gold_message(
                    gold_choice(strategies, last_actions, self_id, rng),
                    rng,
                )
                if p == self_id
                else opponent_message(strategies[p], p, last_actions, rng)
            )
            for p in PLAYERS
        }
        comm_state = {
            "game": "Three-Player Iterated Prisoner's Dilemma",
            "self_id": self_id,
            "players": list(PLAYERS),
            "round": round_idx,
            "max_rounds": MAX_ROUNDS,
            "phase": "communication",
            "scores": dict(total_scores),
            "round_history": list(round_logs),
            "public_chat": messages,
            "rules": {
                "players": 3,
                "payoff": "pairwise PD with T=5,R=3,P=1,S=0",
                "communication": True,
                "decision_format": "[cooperate] or [defect]",
            },
        }
        comm_belief = opponent_belief(strategies, last_actions, self_id)
        comm_choice = gold_choice(strategies, last_actions, self_id, rng)
        comm_out = {
            "belief": comm_belief,
            "reasoning": "Send a credible signal aligned with the intended decision.",
            "action": {"message": messages[self_id]},
        }
        comm_inp = {
            "game_state": comm_state,
            "task": "Exchange a one-line message before the decision phase.",
        }
        examples.append(_row(comm_inp, comm_out, "communication", strategies, round_idx, None))

        gold = comm_choice
        decision_state = {
            **comm_state,
            "phase": "decision",
            "public_chat": messages,
        }
        decision_belief = opponent_belief(strategies, last_actions, self_id)
        decision_out = {
            "belief": decision_belief,
            "reasoning": (
                "Opponents look cooperative; mutual cooperation maximizes pairwise reward."
                if gold == "cooperate"
                else "At least one opponent is likely to defect; protect cumulative score."
            ),
            "action": {"choice": format_choice(gold)},
        }
        decision_inp = {
            "game_state": decision_state,
            "task": "Choose cooperate or defect for this round.",
        }
        opp_actions = {
            p: opponent_action(strategies[p], p, last_actions, rng) for p in PLAYERS if p != self_id
        }
        examples.append(
            _row(decision_inp, decision_out, "decision", strategies, round_idx, {**opp_actions, self_id: gold})
        )

        actions = {self_id: gold, **opp_actions}
        gained = round_scores(actions)
        for p in PLAYERS:
            total_scores[p] += gained[p]
        last_actions = actions
        round_logs.append(
            {
                "round": round_idx,
                "messages": messages,
                "actions": {p: format_choice(actions[p]) for p in PLAYERS},
                "round_scores": gained,
            }
        )

    return examples


def _row(inp, out, phase, strategies, round_idx, actions):
    prompt = (
        "You are playing Three-Player Iterated Prisoner's Dilemma (MindGames).\n"
        "Each round has a communication phase then a simultaneous decision phase.\n"
        "Legal decisions: [cooperate] or [defect]. Payoffs are summed over all pairs.\n\n"
        f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
        "Respond as JSON with keys: belief, reasoning, action."
    )
    meta = {
        "phase": phase,
        "round": round_idx,
        "opponent_strategies": {p: strategies[p] for p in PLAYERS if p != inp["game_state"]["self_id"]},
    }
    if actions is not None:
        meta["joint_actions"] = actions
    return {
        "prompt": prompt,
        "completion": json.dumps(out, ensure_ascii=False),
        "input": inp,
        "output": out,
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="games/ipd/sft.jsonl")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_examples = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n_games):
            strategies = {p: choose_opponent_strategy(rng) for p in PLAYERS}
            for self_id in PLAYERS:
                for ex in play_game(rng, strategies, self_id):
                    row = {k: ex[k] for k in ("prompt", "completion", "input", "output", "meta")}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_examples += 1

    print(f"Wrote {n_examples} examples to {args.out}")


if __name__ == "__main__":
    main()
