"""Parse Secret Mafia observations (MGC / TextArena format)."""

from __future__ import annotations

import json
import re
from typing import Any


def _extract_observation(prompt: str) -> str:
    if "OBSERVATION:" in prompt:
        return prompt.split("OBSERVATION:", 1)[1].strip()
    if "STATE:" in prompt:
        return prompt
    return prompt


def parse_sft_state(prompt: str) -> dict[str, Any] | None:
    if "STATE:" not in prompt:
        return None
    blob = prompt.split("STATE:", 1)[1].split("Respond as JSON", 1)[0].strip()
    try:
        inp = json.loads(blob)
    except json.JSONDecodeError:
        return None
    gs = inp.get("game_state", {})
    priv = inp.get("private_info", {})
    alive = [str(p) for p in gs.get("alive", [])]
    return {
        "format": "sft",
        "self_id": str(priv.get("self_id", "")),
        "self_role": priv.get("self_role"),
        "alive_players": alive,
        "dead_players": [str(p) for p in gs.get("dead", [])],
        "public_chat": gs.get("public_chat", []),
        "vote_history": gs.get("vote_history", []),
        "claims": gs.get("claims", {}),
        "day": gs.get("day", 1),
        "phase": "day_vote",
        "valid_targets": [p for p in alive if p != str(priv.get("self_id", ""))],
        "raw_observation": blob[:2000],
    }


def parse_mgc_observation(prompt: str) -> dict[str, Any]:
    obs = _extract_observation(prompt)
    parsed: dict[str, Any] = {
        "format": "mgc",
        "raw_observation": obs,
        "public_chat": [],
        "vote_history": [],
        "claims": {},
        "dead_players": [],
        "valid_targets": [],
        "phase": "unknown",
    }

    m = re.search(r"You are Player\s+(\d+)", obs, re.I)
    if m:
        parsed["self_id"] = m.group(1)

    m = re.search(r"Your role:\s*(\w+)", obs, re.I)
    if m:
        parsed["self_role"] = m.group(1).capitalize()

    players = re.findall(r"Player\s+(\d+)", obs)
    if players:
        parsed["alive_players"] = sorted(set(players), key=int)

    for dead in re.findall(r"Player\s+(\d+)\s+was killed", obs, re.I):
        parsed["dead_players"].append(dead)
        if "alive_players" in parsed:
            parsed["alive_players"] = [p for p in parsed["alive_players"] if p != dead]

    if re.search(r"Voting phase", obs, re.I):
        parsed["phase"] = "vote"
    elif re.search(r"Night phase", obs, re.I):
        parsed["phase"] = "night"
    elif re.search(r"Day breaks|Discuss", obs, re.I):
        parsed["phase"] = "day_discussion"
    elif re.search(r"investigate", obs, re.I):
        parsed["phase"] = "night_investigate"

    vm = re.search(r"Valid:\s*(\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*)", obs)
    if vm:
        parsed["valid_targets"] = re.findall(r"\[(\d+)\]", vm.group(1))

    for line in obs.splitlines():
        cm = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if cm and cm.group(1) != "-1":
            parsed["public_chat"].append({"speaker": cm.group(1), "text": cm.group(2).strip()})

    for speaker, role in re.findall(
        r"Player\s+(\d+)[^\n]{0,80}?(?:claims?|I am|I'm)\s+(?:the\s+)?(Mafia|Detective|Doctor|Villager|Citizen)",
        obs,
        re.I,
    ):
        parsed["claims"][speaker] = role.capitalize()

    if not parsed.get("valid_targets") and parsed.get("alive_players"):
        me = parsed.get("self_id")
        parsed["valid_targets"] = [p for p in parsed["alive_players"] if p != me]

    parsed.setdefault("self_id", "0")
    parsed.setdefault("alive_players", parsed.get("valid_targets", []))
    parsed.setdefault("day", 1)
    return parsed


def parse_prompt(prompt: str) -> dict[str, Any]:
    sft = parse_sft_state(prompt)
    if sft:
        return sft
    return parse_mgc_observation(prompt)


def infer_required_action_format(parsed: dict[str, Any], observation: str) -> str:
    phase = parsed.get("phase", "")
    if phase == "vote":
        return "Submit vote as [X] where X is a valid player id."
    if phase in ("night", "night_investigate"):
        return "Submit investigation as [Player X] or 'I will investigate Player X'."
    if parsed.get("format") == "sft":
        return "Respond with JSON: belief, reasoning, action {speak, vote}."
    if "submit one vote" in observation.lower():
        return "Submit vote as [X]."
    return "Follow the TextArena format shown in the observation."
