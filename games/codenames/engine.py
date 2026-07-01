"""Codenames rules aligned with MindGames / TextArena (simplified board generator)."""

from __future__ import annotations

import random
import re
from typing import Literal

BOARD_SIZE = 25
RED_COUNT = 9
BLUE_COUNT = 8
NEUTRAL_COUNT = 7
ASSASSIN_COUNT = 1

ROLES = {
    "P0": ("red", "spymaster"),
    "P1": ("red", "operative"),
    "P2": ("blue", "spymaster"),
    "P3": ("blue", "operative"),
}

WORD_POOL = [
    "apple", "bank", "bear", "bell", "bird", "boat", "book", "boot", "bowl",
    "cake", "car", "card", "cat", "chair", "cloud", "coin", "cold", "corn",
    "desk", "dog", "door", "drum", "duck", "dust", "egg", "fan", "field",
    "fire", "fish", "flag", "flame", "flower", "fork", "frog", "game", "gate",
    "glass", "gold", "grass", "hand", "hat", "hill", "horse", "house", "ice",
    "ink", "iron", "jam", "jet", "key", "king", "kite", "lake", "lamp", "leaf",
    "lion", "lock", "map", "milk", "moon", "mouse", "music", "nest", "night",
    "nose", "note", "ocean", "orange", "owl", "paint", "park", "pen", "pie",
    "plane", "plant", "plate", "queen", "rain", "ring", "river", "road", "rock",
    "roof", "rose", "salt", "sand", "school", "seed", "ship", "shoe", "shop",
    "silver", "sky", "snake", "snow", "sock", "song", "star", "stone", "sun",
    "table", "tail", "tank", "tree", "truck", "watch", "water", "wave", "wheel",
    "wind", "window", "wolf", "wood", "worm", "yard", "zebra",
]

CLUE_RE = re.compile(r"\[?\s*([A-Za-z]+)\s+(\d+)\s*\]?", re.IGNORECASE)
GUESS_RE = re.compile(r"\[?\s*(pass|[A-Za-z]+)\s*\]?", re.IGNORECASE)


def new_board(rng: random.Random, starting_team: Literal["red", "blue"] = "red") -> dict:
    words = rng.sample(WORD_POOL, BOARD_SIZE)
    if starting_team == "red":
        counts = {"red": RED_COUNT, "blue": BLUE_COUNT, "neutral": NEUTRAL_COUNT, "assassin": ASSASSIN_COUNT}
    else:
        counts = {"red": BLUE_COUNT, "blue": RED_COUNT, "neutral": NEUTRAL_COUNT, "assassin": ASSASSIN_COUNT}
    labels: list[str] = []
    for label, n in counts.items():
        labels.extend([label] * n)
    rng.shuffle(labels)
    return {
        "words": words,
        "labels": {w: labels[i] for i, w in enumerate(words)},
        "starting_team": starting_team,
    }


def public_board(board: dict, revealed: set[str]) -> list[dict]:
    return [
        {"word": w, "revealed": w in revealed, "color": board["labels"][w] if w in revealed else None}
        for w in board["words"]
    ]


def team_words(board: dict, team: str) -> list[str]:
    return [w for w, label in board["labels"].items() if label == team]


def unrevealed_words(board: dict, revealed: set[str]) -> list[str]:
    return [w for w in board["words"] if w not in revealed]


def is_valid_clue(clue_word: str, board_words: list[str]) -> bool:
    clue = clue_word.lower()
    if not clue.isalpha():
        return False
    for w in board_words:
        wl = w.lower()
        if clue == wl or clue in wl or wl in clue:
            return False
    return True


def parse_clue(text: str) -> tuple[str, int] | None:
    m = CLUE_RE.search(str(text))
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def parse_guess(text: str) -> str | None:
    m = GUESS_RE.search(str(text))
    if not m:
        return None
    return m.group(1).lower()


def format_clue(word: str, count: int) -> str:
    return f"[{word} {count}]"


def format_guess(word: str) -> str:
    return f"[{word}]"


def pick_clue(board: dict, team: str, rng: random.Random) -> tuple[str, int]:
    own = [w for w in team_words(board, team) if w not in {"assassin"}]
    rng.shuffle(own)
    target_words = own[: rng.randint(2, min(3, len(own)))]
    clue_candidates = ["metal", "nature", "animal", "food", "travel", "music", "color", "tool"]
    for clue in rng.sample(clue_candidates, len(clue_candidates)):
        if is_valid_clue(clue, board["words"]):
            return clue, len(target_words)
    return "team", 1


def pick_guess(board: dict, team: str, clue: tuple[str, int], revealed: set[str], rng: random.Random) -> str:
    own = [w for w in team_words(board, team) if w not in revealed]
    if own and rng.random() < 0.75:
        return rng.choice(own)
    hidden = unrevealed_words(board, revealed)
    return rng.choice(hidden) if hidden else "pass"


def operative_belief(board: dict, team: str, clue: tuple[str, int], revealed: set[str]) -> dict[str, float]:
    belief: dict[str, float] = {}
    for w in unrevealed_words(board, revealed):
        belief[w] = 0.7 if w in team_words(board, team) else round(random.Random(hash(w)).uniform(0.01, 0.2), 2)
    return belief
