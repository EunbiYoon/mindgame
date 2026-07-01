"""STARS agent: Guided Generation + ReAct + PAL + validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from stars.agent.belief_state import BeliefState, _norm_player
from stars.agent.code_executor import execute_react_code, validation_code
from stars.parse.observation import infer_required_action_format, parse_prompt
from stars.ollama_client import OllamaClient
from stars.prompts.prompts import (
    FINAL_PROMPT,
    POST_GAME_PROMPT,
    QUESTIONNAIRE,
    REACT_PROMPT,
    SYSTEM_PROMPT,
    VALIDATION_PROMPT,
    format_belief,
    format_code_log,
)
from stars.parse.schemas import Response, parse_react_step, parse_response


@dataclass
class AgentResult:
    response: Response | None
    raw_final: str = ""
    react_steps: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = False
    validation_error: str = ""
    action_text: str = ""


class StarsMafiaAgent:
    def __init__(
        self,
        client: OllamaClient | None = None,
        max_react_steps: int = 3,
        max_retries: int = 2,
    ) -> None:
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

    def post_game_analysis(self, game_key: str, trajectory_summary: str) -> str:
        prompt = POST_GAME_PROMPT.format(summary=trajectory_summary[:3000])
        lesson = self.client.generate(prompt, system=SYSTEM_PROMPT, temperature=0.3)
        lesson = lesson.split("\n")[0].strip()
        self.add_lesson(game_key, lesson)
        return lesson

    def act(
        self,
        prompt: str,
        *,
        game_key: str = "",
        belief: BeliefState | None = None,
    ) -> AgentResult:
        parsed = parse_prompt(prompt)
        obs = parsed.get("raw_observation", prompt)
        if belief is None:
            belief = BeliefState()
        belief.update_from_parsed(parsed)
        for lesson in self.lessons_for_game(game_key):
            belief.add_lesson(lesson)

        react_steps: list[dict[str, Any]] = []
        for _ in range(self.max_react_steps):
            react_prompt = REACT_PROMPT.format(
                belief=format_belief(belief.to_context()),
                observation=obs[:4000],
                questionnaire=QUESTIONNAIRE,
            )
            raw_react = self.client.generate(react_prompt, system=SYSTEM_PROMPT)
            step = parse_react_step(raw_react)
            exec_out = execute_react_code(step.code, belief)
            react_steps.append(
                {
                    "thought": step.thought,
                    "code": step.code,
                    **exec_out,
                }
            )
            if not step.code:
                break

        action_format = infer_required_action_format(parsed, obs)
        lessons_text = "\n".join(f"- {l}" for l in belief.lessons) or "(none)"
        final_prompt = FINAL_PROMPT.format(
            belief=format_belief(belief.to_context()),
            code_log=format_code_log(react_steps),
            observation=obs[:4000],
            lessons=lessons_text,
            action_format=action_format,
        )

        response: Response | None = None
        raw_final = ""
        validation_error = ""
        for attempt in range(self.max_retries + 1):
            raw_final = self.client.generate(final_prompt, system=SYSTEM_PROMPT)
            response = parse_response(raw_final)
            if response is None:
                validation_error = "JSON parse failed"
            else:
                validation_error = self._validate(response, belief, parsed)
                if not validation_error:
                    break
            if attempt < self.max_retries:
                final_prompt = VALIDATION_PROMPT.format(
                    error=validation_error,
                    valid_targets=belief.valid_targets or belief.other_players(),
                )

        valid = response is not None and not validation_error
        action_text = self._render_action(response, belief, parsed) if response else raw_final
        if response:
            response.code_observations = [
                s.get("stdout") or json.dumps(s.get("result", {}), ensure_ascii=False)
                for s in react_steps
                if s.get("stdout") or s.get("result")
            ]

        return AgentResult(
            response=response,
            raw_final=raw_final,
            react_steps=react_steps,
            valid=valid,
            validation_error=validation_error,
            action_text=action_text,
        )

    def _validate(self, response: Response, belief: BeliefState, parsed: dict[str, Any]) -> str:
        phase = parsed.get("phase", "")
        vote = _norm_player(response.action.vote)
        raw = response.action.raw.strip()
        valid_targets = belief.valid_targets or belief.other_players()

        if parsed.get("format") == "sft":
            if not vote:
                return "SFT format requires action.vote"
            check = validation_code(vote, valid_targets, belief.alive_players)
            if not check["valid"]:
                return f"vote {vote!r} not in {valid_targets}"
            return ""

        if phase == "vote" or "submit one vote" in parsed.get("raw_observation", "").lower():
            target = vote or self._extract_vote_from_raw(raw)
            if not target:
                return "vote phase requires action.vote or action.raw with [X]"
            check = validation_code(target, valid_targets, belief.alive_players)
            if not check["valid"]:
                return f"vote {target!r} not in {valid_targets}"

        return ""

    @staticmethod
    def _extract_vote_from_raw(raw: str) -> str:
        m = re.search(r"\[(\d+)\]", raw)
        if m:
            return m.group(1)
        m = re.search(r"vote:?\s*player\s*(\d+)", raw, re.I)
        if m:
            return m.group(1)
        return ""

    def _render_action(self, response: Response, belief: BeliefState, parsed: dict[str, Any]) -> str:
        if parsed.get("format") == "sft":
            out = {
                "belief": {p: {"mafia": v} for p, v in belief.p_mafia.items()},
                "reasoning": response.reasoning,
                "action": {
                    "speak": response.action.speak,
                    "vote": response.action.vote,
                },
            }
            return json.dumps(out, ensure_ascii=False)

        if response.action.raw:
            return response.action.raw
        if response.action.vote and parsed.get("phase") == "vote":
            return f"[{response.action.vote}]"
        if response.action.speak:
            return response.action.speak
        return response.reasoning
