"""STARS agent for Blotto / IPD / Codenames (ReAct + PAL-lite)."""

from __future__ import annotations

import io
import json
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from eval.metrics import extract_json
from games.blotto.engine import is_valid_allocation, parse_allocation
from games.codenames.engine import is_valid_clue, parse_clue, parse_guess
from games.ipd.engine import parse_choice
from stars.parse.game_observation import parse_game_prompt, required_action_format
from stars.prompts.game_prompts import (
    FINAL_PROMPT,
    GAME_QUESTIONNAIRE,
    GAME_SYSTEM,
    POST_GAME_PROMPT,
    REACT_PROMPT,
    VALIDATION_PROMPT,
)
from stars.ollama_client import OllamaClient
from stars.prompts.prompts import format_code_log


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass
class GameNotes:
    game: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "parsed": self.parsed,
            "notes": self.notes,
            "lessons": list(self.lessons),
        }

    def apply_code_result(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        for key, value in result.items():
            self.notes[key] = value


def execute_game_code(code: str, notes: GameNotes) -> dict[str, Any]:
    if not code or not code.strip():
        return {"stdout": "", "error": "empty code", "result": {}}

    namespace: dict[str, Any] = {
        "notes": notes.notes,
        "state": notes.to_context(),
        "parsed": notes.parsed,
        "result": {},
    }
    stdout = io.StringIO()
    error = ""
    try:
        with redirect_stdout(stdout):
            exec(code, {"__builtins__": _SAFE_BUILTINS}, namespace)
    except Exception:
        error = traceback.format_exc(limit=3)

    result = namespace.get("result", {})
    if not isinstance(result, dict):
        result = {"value": result}
    notes.apply_code_result(result)
    return {
        "stdout": stdout.getvalue().strip(),
        "error": error,
        "result": result,
    }


def _parse_react_step(text: str) -> tuple[str, str]:
    import re

    thought = ""
    code = ""
    t = re.search(r"(?i)thought:\s*(.+?)(?=action:|```|$)", text, flags=re.S)
    if t:
        thought = t.group(1).strip()
    c = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if c:
        code = c.group(1).strip()
    return thought, code


@dataclass
class GameAgentResult:
    action_text: str = ""
    valid: bool = False
    validation_error: str = ""
    react_steps: list[dict[str, Any]] = field(default_factory=list)


class StarsGameAgent:
    def __init__(
        self,
        game: str,
        client: OllamaClient | None = None,
        max_react_steps: int = 3,
        max_retries: int = 2,
    ) -> None:
        self.game = game
        self.client = client or OllamaClient()
        self.max_react_steps = max_react_steps
        self.max_retries = max_retries
        self._game_lessons: dict[str, list[str]] = {}

    def lessons_for_game(self, game_key: str) -> list[str]:
        return self._game_lessons.get(game_key, [])

    def add_lesson(self, game_key: str, lesson: str) -> None:
        lesson = lesson.strip()
        if not lesson:
            return
        bucket = self._game_lessons.setdefault(game_key, [])
        if lesson not in bucket:
            bucket.append(lesson)

    def post_game_analysis(self, game_key: str, trajectory_summary: str, *, game_title: str) -> str:
        prompt = POST_GAME_PROMPT.format(summary=trajectory_summary[:3000], game_title=game_title)
        lesson = self.client.generate(prompt, system=GAME_SYSTEM[self.game], temperature=0.3)
        lesson = lesson.split("\n")[0].strip()
        self.add_lesson(game_key, lesson)
        return lesson

    def act(self, prompt: str, *, game_key: str = "", meta: dict[str, Any] | None = None) -> GameAgentResult:
        parsed = parse_game_prompt(self.game, prompt, meta)
        notes = GameNotes(game=self.game, parsed=parsed)
        for lesson in self.lessons_for_game(game_key):
            notes.lessons.append(lesson)
        obs = parsed.get("raw_observation", prompt)
        system = GAME_SYSTEM[self.game]
        questionnaire = GAME_QUESTIONNAIRE[self.game]
        action_format = required_action_format(self.game, parsed)

        react_steps: list[dict[str, Any]] = []
        for _ in range(self.max_react_steps):
            react_prompt = REACT_PROMPT.format(
                notes=json.dumps(notes.to_context(), ensure_ascii=False, indent=2),
                observation=obs[:4000],
                questionnaire=questionnaire,
            )
            raw_react = self.client.generate(react_prompt, system=system)
            thought, code = _parse_react_step(raw_react)
            exec_out = execute_game_code(code, notes)
            react_steps.append({"thought": thought, "code": code, **exec_out})
            if not code:
                break

        lessons_text = "\n".join(f"- {l}" for l in notes.lessons) or "(none)"
        final_prompt = FINAL_PROMPT.format(
            notes=json.dumps(notes.to_context(), ensure_ascii=False, indent=2),
            code_log=format_code_log(react_steps),
            observation=obs[:4000],
            lessons=lessons_text,
            action_format=action_format,
        )

        action_text = ""
        validation_error = ""
        for attempt in range(self.max_retries + 1):
            action_text = self.client.generate(final_prompt, system=system).strip()
            validation_error = self._validate(action_text, parsed)
            if not validation_error:
                break
            if attempt < self.max_retries:
                final_prompt = VALIDATION_PROMPT.format(
                    error=validation_error,
                    action_format=action_format,
                )

        return GameAgentResult(
            action_text=action_text,
            valid=not validation_error,
            validation_error=validation_error,
            react_steps=react_steps,
        )

    def _validate(self, action_text: str, parsed: dict[str, Any]) -> str:
        if self.game == "blotto":
            obj = extract_json(action_text)
            text = str(obj["action"]["allocation"]) if isinstance(obj, dict) and isinstance(obj.get("action"), dict) else action_text
            alloc = parse_allocation(text)
            if alloc is None:
                return "invalid allocation format"
            if not is_valid_allocation(alloc):
                return "allocation must sum to 20 with non-negative field units"
            return ""

        if self.game == "ipd":
            if parsed.get("phase") == "communication":
                return "" if action_text.strip() else "communication requires a message"
            if _ipd_decision_valid(action_text):
                return ""
            return "decision phase requires [pid cooperate] or [pid defect] per opponent"

        if self.game == "codenames":
            role = parsed.get("role", "operative")
            obj = extract_json(action_text)
            text = action_text
            if isinstance(obj, dict) and isinstance(obj.get("action"), dict):
                key = "clue" if role == "spymaster" else "guess"
                if obj["action"].get(key):
                    text = str(obj["action"][key])
            if role == "spymaster":
                clue = parse_clue(text)
                if clue is None:
                    return "spymaster requires [word number] clue"
                word, n = clue
                board = parsed.get("board_words", [])
                if n < 1 or not is_valid_clue(word, board):
                    return "invalid spymaster clue"
                return ""
            guess = parse_guess(text)
            if guess in ("pass",):
                return ""
            board = parsed.get("board_words", [])
            if guess and guess in board:
                return ""
            return "operative requires [word] or [pass]"

        return ""


def _ipd_decision_valid(text: str) -> bool:
    import re

    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("action"), dict):
        choice = obj["action"].get("choice")
        if parse_choice(str(choice)) in ("cooperate", "defect"):
            return True
    tokens = re.findall(r"\[\s*\d+\s+(cooperate|defect)\s*\]", text, flags=re.I)
    if len(tokens) >= 2:
        return True
    if parse_choice(text) in ("cooperate", "defect"):
        return True
    return False


def dry_run_action(game: str, prompt: str, meta: dict[str, Any] | None = None) -> str:
    parsed = parse_game_prompt(game, prompt, meta)
    if game == "blotto":
        return "[A7 B7 C6]"
    if game == "ipd":
        if parsed.get("phase") == "communication":
            return "I plan to cooperate this round if others do the same."
        return "[1 cooperate] [2 cooperate]"
    if game == "codenames":
        if parsed.get("role") == "spymaster":
            return "[team 2]"
        return "[pass]"
    return ""
