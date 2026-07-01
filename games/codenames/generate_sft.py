#!/usr/bin/env python3
"""Generate Codenames SFT data (MindGames Spymaster / Operative turns)."""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from engine import (  # noqa: E402
    ROLES,
    format_clue,
    format_guess,
    is_valid_clue,
    new_board,
    operative_belief,
    parse_clue,
    pick_clue,
    pick_guess,
    public_board,
    team_words,
)


def play_game(rng: random.Random) -> list[dict]:
    board = new_board(rng)
    revealed: set[str] = set()
    examples = []
    turn_no = 0
    active_team = board["starting_team"]
    current_clue: tuple[str, int] | None = None

    order = ["P0", "P1", "P2", "P3"]
    idx = 0
    while len(revealed) < len(board["words"]) - 1 and turn_no < 30:
        player = order[idx % len(order)]
        team, role = ROLES[player]
        if team != active_team:
            idx += 1
            continue

        turn_no += 1
        if role == "spymaster":
            clue_word, clue_n = pick_clue(board, team, rng)
            current_clue = (clue_word, clue_n)
            belief = {
                "operative_likely_targets": team_words(board, team)[: clue_n + 1],
                "avoid": [w for w, lab in board["labels"].items() if lab in ("assassin", "blue" if team == "red" else "red")][:3],
            }
            reasoning = (
                f"Give a one-word clue related to {clue_n} team words while avoiding board words and the assassin."
            )
            action = {"clue": format_clue(clue_word, clue_n)}
            game_state = {
                "game": "Codenames",
                "self_id": player,
                "role": role,
                "team": team,
                "turn": turn_no,
                "board": public_board(board, revealed),
                "hidden_labels": board["labels"],
                "revealed": sorted(revealed),
                "rules": {
                    "clue_format": "[word n]",
                    "guess_format": "[word] or [pass]",
                    "players": 4,
                },
            }
            task = "Give a legal one-word clue and number for your operative."
        else:
            if current_clue is None:
                idx += 1
                continue
            guess = pick_guess(board, team, current_clue, revealed, rng)
            belief = operative_belief(board, team, current_clue, revealed)
            reasoning = f"Interpret clue {format_clue(*current_clue)} and pick the best unrevealed team word."
            action = {"guess": format_guess(guess)}
            game_state = {
                "game": "Codenames",
                "self_id": player,
                "role": role,
                "team": team,
                "turn": turn_no,
                "board": public_board(board, revealed),
                "current_clue": format_clue(*current_clue),
                "revealed": sorted(revealed),
                "rules": {
                    "clue_format": "[word n]",
                    "guess_format": "[word] or [pass]",
                    "players": 4,
                },
            }
            task = "Guess a board word or pass based on the spymaster clue."
            if guess != "pass" and guess in board["labels"]:
                revealed.add(guess)
                label = board["labels"][guess]
                if label == "assassin":
                    break
                if label != team:
                    active_team = "blue" if team == "red" else "red"
                    current_clue = None
            else:
                active_team = "blue" if team == "red" else "red"
                current_clue = None

        inp = {"game_state": game_state, "task": task}
        out = {"belief": belief, "reasoning": reasoning, "action": action}
        prompt = (
            "You are playing Codenames (MindGames).\n"
            "Teams use constrained signaling: Spymaster gives [word n], Operative guesses [word] or [pass].\n\n"
            f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
            "Respond as JSON with keys: belief, reasoning, action."
        )
        meta = {
            "role": role,
            "team": team,
            "turn": turn_no,
            "gold_action": action,
        }
        if role == "spymaster":
            meta["clue_valid"] = is_valid_clue(parse_clue(action["clue"])[0], board["words"]) if parse_clue(action["clue"]) else False
        examples.append(
            {
                "prompt": prompt,
                "completion": json.dumps(out, ensure_ascii=False),
                "input": inp,
                "output": out,
                "meta": meta,
            }
        )
        idx += 1

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="games/codenames/sft.jsonl")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_examples = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n_games):
            for ex in play_game(rng):
                row = {k: ex[k] for k in ("prompt", "completion", "input", "output", "meta")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_examples += 1

    print(f"Wrote {n_examples} examples to {args.out}")


if __name__ == "__main__":
    main()
