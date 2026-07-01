"""Convert mindgameschallenge/MGC2025 trajectories to LoRA SFT jsonl."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

DATASET_ID = "mindgameschallenge/MGC2025"

GAMES: dict[str, dict[str, str]] = {
    "mafia": {
        "hf_config": "secretmafia",
        "title": "Secret Mafia",
        "env_names": ("SecretMafia-v0",),
    },
    "blotto": {
        "hf_config": "colonelblotto",
        "title": "Colonel Blotto",
        "env_names": ("ColonelBlotto-v0",),
    },
    "ipd": {
        "hf_config": "threeplayeripd",
        "title": "Three-Player IPD",
        "env_names": ("ThreePlayerIPD-v0",),
    },
    "codenames": {
        "hf_config": "codenames",
        "title": "Codenames",
        "env_names": ("Codenames-v0",),
    },
}

PROMPTS: dict[str, str] = {
    "mafia": (
        "You are playing Secret Mafia (MindGames / TextArena).\n"
        "Read the observation and respond with your next action.\n"
        "Follow the game format shown in the observation.\n\n"
        "OBSERVATION:\n"
    ),
    "blotto": (
        "You are Commander in Colonel Blotto (MindGames / TextArena).\n"
        "Allocate units across fields A, B, C. Format: [A5 B10 C5]\n\n"
        "OBSERVATION:\n"
    ),
    "ipd": (
        "You are playing Three-Player Iterated Prisoner's Dilemma (MindGames / TextArena).\n"
        "Respond in the format required by the current phase (chat or [pid cooperate/defect]).\n\n"
        "OBSERVATION:\n"
    ),
    "codenames": (
        "You are playing Codenames (MindGames / TextArena).\n"
        "Spymaster: [word n]. Operative: [word] or [pass].\n\n"
        "OBSERVATION:\n"
    ),
}


def parse_observations(raw: Any) -> list[tuple[str, str]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return []
    steps: list[tuple[str, str, str]] = []
    for ts, step in raw.items():
        if not isinstance(step, dict):
            continue
        obs = str(step.get("observation", "")).strip()
        act = str(step.get("action", "")).strip()
        if obs and act:
            steps.append((str(ts), obs, act))
    steps.sort(key=lambda x: x[0])
    return [(obs, act) for _, obs, act in steps]


def detect_ipd_phase(observation: str) -> str:
    low = observation.lower()
    if "submit your decision" in low or "submit your decisions" in low:
        return "decision"
    return "communication"


def detect_codenames_role(observation: str) -> str:
    low = observation.lower()
    if "spymaster" in low and "operative" not in low[:120]:
        return "spymaster"
    if "operative" in low:
        return "operative"
    if "give a one-word clue" in low or "one-word clue" in low:
        return "spymaster"
    return "unknown"


def wrap_completion(game: str, action: str, observation: str) -> str:
    """Raw TextArena action as completion (paper-style imitation)."""
    return action.strip()


def trajectory_examples(game: str, row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    turns = parse_observations(row.get("observations"))
    if not turns:
        return

    game_id = row.get("game_id")
    for turn_idx, (observation, action) in enumerate(turns):
        meta: dict[str, Any] = {
            "source": "mgc2025",
            "game": game,
            "game_id": game_id,
            "player_game_id": row.get("player_game_id"),
            "env_name": row.get("env_name"),
            "model_name": row.get("model_name"),
            "player_id": row.get("player_id"),
            "turn": turn_idx,
            "num_turns": row.get("num_turns"),
            "rewards": row.get("rewards"),
            "status": row.get("status"),
        }
        if game == "ipd":
            meta["phase"] = detect_ipd_phase(observation)
        if game == "codenames":
            meta["role"] = detect_codenames_role(observation)

        completion = wrap_completion(game, action, observation)
        prompt = PROMPTS[game] + observation
        yield {
            "prompt": prompt,
            "completion": completion,
            "meta": meta,
        }


def is_test_row(row_id: Any, *, test_frac: float, seed: int) -> bool:
    if test_frac <= 0:
        return False
    key = f"{seed}:{row_id}"
    bucket = sum(ord(c) for c in str(key)) % 10_000
    return bucket < int(test_frac * 10_000)


def norm_action(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def action_exact_match(pred: str, gold: str) -> bool:
    return norm_action(pred) == norm_action(gold)


def action_contains_match(pred: str, gold: str) -> bool:
    p, g = norm_action(pred), norm_action(gold)
    if not g:
        return False
    return g in p or p in g
