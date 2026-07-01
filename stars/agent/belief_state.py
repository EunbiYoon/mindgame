"""Python belief state for STARS Mafia agent (PAL memory)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _norm_player(pid: str) -> str:
    s = str(pid).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


@dataclass
class BeliefState:
    self_id: str = ""
    self_role: str | None = None
    alive_players: list[str] = field(default_factory=list)
    dead_players: list[str] = field(default_factory=list)
    vote_history: list[dict[str, Any]] = field(default_factory=list)
    p_mafia: dict[str, float] = field(default_factory=dict)
    trust_score: dict[str, float] = field(default_factory=dict)
    claims: dict[str, str] = field(default_factory=dict)
    public_chat: list[dict[str, str]] = field(default_factory=list)
    phase: str = "unknown"
    valid_targets: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    day: int = 1

    def other_players(self) -> list[str]:
        me = _norm_player(self.self_id)
        return [p for p in self.alive_players if _norm_player(p) != me]

    def init_uniform_beliefs(self) -> None:
        others = self.other_players()
        if not others:
            return
        prior = 1.0 / len(others)
        for p in others:
            self.p_mafia.setdefault(_norm_player(p), prior)
            self.trust_score.setdefault(_norm_player(p), 0.5)

    def update_from_parsed(self, parsed: dict[str, Any]) -> None:
        if parsed.get("self_id"):
            self.self_id = _norm_player(parsed["self_id"])
        if parsed.get("self_role"):
            self.self_role = parsed["self_role"]
        if parsed.get("alive_players"):
            self.alive_players = [_norm_player(p) for p in parsed["alive_players"]]
        if parsed.get("dead_players"):
            self.dead_players = [_norm_player(p) for p in parsed["dead_players"]]
        if parsed.get("phase"):
            self.phase = parsed["phase"]
        if parsed.get("valid_targets"):
            self.valid_targets = [_norm_player(p) for p in parsed["valid_targets"]]
        if parsed.get("claims"):
            self.claims = {_norm_player(k): v for k, v in parsed["claims"].items()}
        if parsed.get("public_chat"):
            self.public_chat.extend(parsed["public_chat"])
        if parsed.get("vote_history"):
            self.vote_history.extend(parsed["vote_history"])
        if parsed.get("day"):
            self.day = int(parsed["day"])
        self.init_uniform_beliefs()

    def apply_code_result(self, result: dict[str, Any]) -> None:
        if "p_mafia" in result and isinstance(result["p_mafia"], dict):
            for k, v in result["p_mafia"].items():
                self.p_mafia[_norm_player(k)] = float(v)
        if "trust_score" in result and isinstance(result["trust_score"], dict):
            for k, v in result["trust_score"].items():
                self.trust_score[_norm_player(k)] = float(v)
        if "claims" in result and isinstance(result["claims"], dict):
            for k, v in result["claims"].items():
                self.claims[_norm_player(k)] = str(v)
        if "vote_history" in result and isinstance(result["vote_history"], list):
            self.vote_history.extend(result["vote_history"])
        if "alive_players" in result and isinstance(result["alive_players"], list):
            self.alive_players = [_norm_player(p) for p in result["alive_players"]]

    def add_lesson(self, lesson: str) -> None:
        lesson = lesson.strip()
        if lesson and lesson not in self.lessons:
            self.lessons.append(lesson)

    def to_context(self) -> dict[str, Any]:
        return {
            "self_id": self.self_id,
            "self_role": self.self_role,
            "alive_players": self.alive_players,
            "dead_players": self.dead_players,
            "vote_history": self.vote_history,
            "p_mafia": self.p_mafia,
            "trust_score": self.trust_score,
            "claims": self.claims,
            "public_chat": self.public_chat[-20:],
            "phase": self.phase,
            "valid_targets": self.valid_targets,
            "day": self.day,
            "lessons_learned": self.lessons,
        }

    def top_suspect(self) -> str | None:
        candidates = {
            k: v for k, v in self.p_mafia.items() if _norm_player(k) != _norm_player(self.self_id)
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)
