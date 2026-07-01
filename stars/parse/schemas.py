"""Pydantic schemas for STARS guided generation (reasoning + action)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MafiaAction(BaseModel):
    speak: str = Field(default="", description="Public speech for day discussion")
    vote: str = Field(default="", description="Vote target player id")
    raw: str = Field(default="", description="Raw TextArena action when applicable")

    @field_validator("vote")
    @classmethod
    def strip_vote(cls, v: str) -> str:
        v = str(v or "").strip()
        m = re.search(r"\[?\s*(\d+)\s*\]?", v)
        return m.group(1) if m else v


class Response(BaseModel):
    reasoning: str
    action: MafiaAction
    code_observations: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class ReActStep(BaseModel):
    thought: str = ""
    code: str = ""


def parse_response(text: str) -> Response | None:
    """Parse LLM output into Response; tolerate fenced JSON."""
    if not text:
        return None
    blob = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, flags=re.S)
    if fence:
        blob = fence.group(1)
    else:
        m = re.search(r"\{.*\}", blob, flags=re.S)
        if m:
            blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    try:
        return Response.model_validate(data)
    except Exception:
        return None


def parse_react_step(text: str) -> ReActStep:
    thought = ""
    code = ""
    t = re.search(r"(?i)thought:\s*(.+?)(?=action:|```|$)", text, flags=re.S)
    if t:
        thought = t.group(1).strip()
    c = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if c:
        code = c.group(1).strip()
    return ReActStep(thought=thought, code=code)
