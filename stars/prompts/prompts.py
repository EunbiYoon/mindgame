"""Prompt templates for STARS Mafia agent."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are STARS, a Secret Mafia agent (MindGames Efficient division approach).
You use NO fine-tuning. You reason step-by-step, write Python for belief updates (PAL),
and output structured JSON actions.

Rules:
- Track mafia probability, trust, vote history in Python variables — not in prose memory.
- Separate reasoning from action.
- Never invent players outside alive_players / valid_targets.
- For votes, pick exactly one valid target id.
"""


QUESTIONNAIRE = """Answer these before acting:
1. Who is most suspicious and why?
2. Who defended whom? Does that pattern suggest mafia pairing?
3. What are current P(player=mafia) estimates?
4. Which claims are inconsistent with chat / votes?
5. What is the best speech + vote given your role?
"""


REACT_PROMPT = """You are in a ReAct step. Write:
Thought: <brief analysis>
Action:
```python
# Update belief state. Set `result` dict with keys like p_mafia, trust_score.
# You may read: alive_players, vote_history, p_mafia, trust_score, claims, state
result = {{}}
```

Belief context:
{belief}

Observation:
{observation}

{questionnaire}
"""


FINAL_PROMPT = """Produce the final move as JSON matching this schema:
{{
  "reasoning": "string",
  "action": {{
    "speak": "public message (day) or empty",
    "vote": "player id for vote phase, else empty",
    "raw": "full TextArena action string if observation requires [X] format"
  }},
  "code_observations": ["summary of code execution results"]
}}

Required action format: {action_format}

Belief context:
{belief}

Code execution log:
{code_log}

Observation:
{observation}

Lessons from earlier turns in this game:
{lessons}

Respond with JSON only.
"""


VALIDATION_PROMPT = """The previous action failed validation: {error}
Valid targets: {valid_targets}
Regenerate JSON with a valid action only.
"""


def format_belief(ctx: dict[str, Any]) -> str:
    return json.dumps(ctx, ensure_ascii=False, indent=2)


def format_code_log(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "(no code executed)"
    lines = []
    for i, s in enumerate(steps, 1):
        lines.append(f"Step {i} thought: {s.get('thought', '')[:300]}")
        if s.get("stdout"):
            lines.append(f"  stdout: {s['stdout'][:400]}")
        if s.get("error"):
            lines.append(f"  error: {s['error'][:200]}")
        if s.get("result"):
            lines.append(f"  result: {json.dumps(s.get('result'), ensure_ascii=False)[:400]}")
    return "\n".join(lines)


POST_GAME_PROMPT = """Summarize one lesson learned from this Secret Mafia trajectory for future games.
One sentence only. Focus on vote patterns, claim consistency, or mafia tells.

Trajectory summary:
{summary}
"""
