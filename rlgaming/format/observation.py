"""RLGaming context optimization: compact [GAME] / [PRIVATE] phase blocks."""

from __future__ import annotations

import re
from typing import Any

from mgc2025_sft.lib import detect_codenames_role, detect_ipd_phase
from stars.parse.game_observation import parse_blotto, parse_codenames, parse_ipd
from stars.parse.observation import parse_prompt as parse_mafia_prompt
from rlgaming.format.prompts import FORMAT_PROMPT


def _dedupe_chat(chat: list[dict[str, Any]], max_lines: int = 12) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for entry in chat:
        speaker = str(entry.get("speaker", ""))
        text = re.sub(r"\s+", " ", str(entry.get("text", ""))).strip()
        if not text:
            continue
        key = (speaker, text[:240])
        if key in seen:
            continue
        seen.add(key)
        out.append({"speaker": speaker, "text": text[:400]})
    return out[-max_lines:]


def _format_votes(vote_history: list[dict[str, Any]]) -> str:
    if not vote_history:
        return ""
    last = vote_history[-1]
    votes = last.get("votes") or {}
    if not votes:
        return ""
    parts = [f"{voter}->{target}" for voter, target in sorted(votes.items())]
    return f"day {last.get('day', '?')}: " + ", ".join(parts)


def _extract_observation(prompt: str) -> str:
    if "OBSERVATION:" in prompt:
        return prompt.split("OBSERVATION:", 1)[1].strip()
    if "STATE:" in prompt:
        return prompt
    return prompt


def build_mafia_context(parsed: dict[str, Any]) -> str:
    lines = ["[GAME]"]
    phase = parsed.get("phase", "unknown")
    lines.append(f"phase: {phase}")
    if parsed.get("day"):
        lines.append(f"day: {parsed['day']}")

    alive = parsed.get("alive_players") or []
    if alive:
        lines.append(f"alive_players: {', '.join(str(p) for p in alive)}")

    dead = parsed.get("dead_players") or []
    if dead:
        lines.append(f"dead_players: {', '.join(str(p) for p in dead)}")

    claims = parsed.get("claims") or {}
    if claims:
        claim_str = ", ".join(f"{p}={role}" for p, role in sorted(claims.items()))
        lines.append(f"role_claims: {claim_str}")

    chat = _dedupe_chat(parsed.get("public_chat") or [])
    if chat:
        lines.append("public_history:")
        for c in chat:
            lines.append(f"  [{c['speaker']}] {c['text']}")

    vote_line = _format_votes(parsed.get("vote_history") or [])
    if vote_line:
        lines.append(f"last_votes: {vote_line}")

    targets = parsed.get("valid_targets") or []
    if targets:
        lines.append(f"valid_targets: {', '.join(str(t) for t in targets)}")

    lines.append("")
    lines.append("[PRIVATE]")
    if parsed.get("self_id") is not None:
        lines.append(f"my_id: {parsed['self_id']}")
    if parsed.get("self_role"):
        lines.append(f"my_role: {parsed['self_role']}")
    return "\n".join(lines)


def build_blotto_context(parsed: dict[str, Any]) -> str:
    obs = parsed.get("raw_observation", "")
    lines = [
        "[GAME]",
        f"round: {parsed.get('round_idx', 1)}",
        f"fields: {', '.join(parsed.get('fields', ['A', 'B', 'C']))}",
        f"units_to_allocate: {parsed.get('units', 20)}",
    ]
    for label in ("Commander Alpha", "Commander Beta"):
        m = re.search(rf"{re.escape(label)} allocated:\s*(.+)", obs, re.I)
        if m:
            lines.append(f"last_{label.lower().replace(' ', '_')}: {m.group(1).strip()}")
    lines.extend(["", "[PRIVATE]", "my_role: Commander"])
    return "\n".join(lines)


def build_ipd_context(parsed: dict[str, Any]) -> str:
    lines = [
        "[GAME]",
        f"phase: {parsed.get('phase', 'unknown')}",
        "players: 0, 1, 2",
        "payoff: T=5 R=3 P=1 S=0",
    ]
    lines.extend(["", "[PRIVATE]", f"my_id: {parsed.get('self_id', '0')}"])
    return "\n".join(lines)


def build_codenames_context(parsed: dict[str, Any]) -> str:
    lines = [
        "[GAME]",
        f"role: {parsed.get('role', 'unknown')}",
    ]
    board = parsed.get("board_words") or []
    if board:
        lines.append(f"board_words: {', '.join(board[:25])}")
    lines.extend(["", "[PRIVATE]", "team: from observation"])
    return "\n".join(lines)


def build_compact_context(game: str, parsed: dict[str, Any]) -> str:
    if game == "mafia":
        return build_mafia_context(parsed)
    if game == "blotto":
        return build_blotto_context(parsed)
    if game == "ipd":
        return build_ipd_context(parsed)
    if game == "codenames":
        return build_codenames_context(parsed)
    return _extract_observation(parsed.get("raw_observation", ""))[:2000]


def parse_for_game(game: str, raw_prompt: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    if game == "mafia":
        parsed = parse_mafia_prompt(raw_prompt)
        parsed["raw_observation"] = _extract_observation(raw_prompt)
        return parsed
    if game == "blotto":
        return parse_blotto(raw_prompt)
    if game == "ipd":
        return parse_ipd(raw_prompt, phase=meta.get("phase"))
    if game == "codenames":
        return parse_codenames(raw_prompt, role=meta.get("role"))
    return {"raw_observation": _extract_observation(raw_prompt)}


def build_rlgaming_prompt(raw_prompt: str, game: str, meta: dict[str, Any] | None = None) -> str:
    parsed = parse_for_game(game, raw_prompt, meta)
    context = build_compact_context(game, parsed)
    return FORMAT_PROMPT.format(context=context)
