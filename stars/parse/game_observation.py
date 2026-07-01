"""Parse Blotto / IPD / Codenames observations for STARS agent."""

from __future__ import annotations

import re
from typing import Any

from mgc2025_sft.lib import detect_codenames_role, detect_ipd_phase


def _extract_observation(prompt: str) -> str:
    if "OBSERVATION:" in prompt:
        return prompt.split("OBSERVATION:", 1)[1].strip()
    return prompt


def parse_blotto(prompt: str) -> dict[str, Any]:
    obs = _extract_observation(prompt)
    parsed: dict[str, Any] = {
        "game": "blotto",
        "raw_observation": obs,
        "round_idx": 1,
        "fields": ["A", "B", "C"],
        "units": 20,
    }
    m = re.search(r"Round\s+(\d+)", obs, re.I)
    if m:
        parsed["round_idx"] = int(m.group(1))
    m = re.search(r"Units to allocate:\s*(\d+)", obs, re.I)
    if m:
        parsed["units"] = int(m.group(1))
    return parsed


def parse_ipd(prompt: str, *, phase: str | None = None) -> dict[str, Any]:
    obs = _extract_observation(prompt)
    parsed: dict[str, Any] = {
        "game": "ipd",
        "raw_observation": obs,
        "phase": phase or detect_ipd_phase(obs),
        "self_id": "0",
        "opponents": ["1", "2"],
    }
    m = re.search(r"You are Player\s+(\d+)", obs, re.I)
    if m:
        parsed["self_id"] = m.group(1)
    return parsed


def parse_codenames(prompt: str, *, role: str | None = None) -> dict[str, Any]:
    obs = _extract_observation(prompt)
    parsed: dict[str, Any] = {
        "game": "codenames",
        "raw_observation": obs,
        "role": role or detect_codenames_role(obs),
        "board_words": [],
    }
    in_board = False
    for line in obs.splitlines():
        if "codenames words" in line.lower():
            in_board = True
            continue
        if not in_board:
            continue
        token = line.strip().split()
        if token:
            parsed["board_words"].append(token[0].lower())
    return parsed


def parse_game_prompt(game: str, prompt: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    if game == "blotto":
        return parse_blotto(prompt)
    if game == "ipd":
        return parse_ipd(prompt, phase=meta.get("phase"))
    if game == "codenames":
        return parse_codenames(prompt, role=meta.get("role"))
    raise ValueError(f"Unsupported game: {game}")


def required_action_format(game: str, parsed: dict[str, Any]) -> str:
    if game == "blotto":
        units = parsed.get("units", 20)
        return f"Allocate exactly {units} units across A, B, C as [A# B# C#]."
    if game == "ipd":
        if parsed.get("phase") == "communication":
            return "Free-form chat message for the communication phase."
        opp = parsed.get("opponents", ["1", "2"])
        tokens = " ".join(f"[{pid} cooperate] or [{pid} defect]" for pid in opp)
        return f"Decision phase: submit one token per opponent, e.g. {tokens}."
    if game == "codenames":
        role = parsed.get("role", "operative")
        if role == "spymaster":
            return "Spymaster clue as [word number], e.g. [ocean 2]."
        return "Operative guess as [word] or [pass]."
    return "Follow the TextArena format in the observation."
