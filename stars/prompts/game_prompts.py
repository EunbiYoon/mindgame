"""Prompt templates for STARS agents on Blotto / IPD / Codenames."""

from __future__ import annotations

GAME_SYSTEM = {
    "blotto": """You are STARS playing Colonel Blotto (MindGames Efficient approach).
Use ReAct + Python to analyze opponent allocations, then output the next move.
No fine-tuning. Follow TextArena format exactly.""",
    "ipd": """You are STARS playing Three-Player IPD (MindGames Efficient approach).
Use ReAct + Python to track cooperation/defection patterns, then output the next move.
No fine-tuning. Match the phase format (chat or decision tokens).""",
    "codenames": """You are STARS playing Codenames (MindGames Efficient approach).
Use ReAct + Python to track clues, guesses, and board state, then output the next move.
No fine-tuning. Follow TextArena clue/guess format.""",
}

GAME_QUESTIONNAIRE = {
    "blotto": """1. What did each commander allocate last round?
2. Which fields are contested?
3. What allocation maximizes win probability given 20 units?
4. Output the next allocation.""",
    "ipd": """1. What phase is this (communication or decision)?
2. What did opponents do in prior rounds?
3. Should you cooperate or defect with each opponent?
4. Output the required action format.""",
    "codenames": """1. Are you spymaster or operative?
2. What clues were given and which words were guessed?
3. Which words remain for your team?
4. Output the next clue or guess.""",
}

REACT_PROMPT = """You are in a ReAct step. Write:
Thought: <brief analysis>
Action:
```python
# Analyze the game. Set `result` dict with notes, scores, or plans.
result = {{}}
```

Game notes:
{notes}

Observation:
{observation}

{questionnaire}
"""

FINAL_PROMPT = """Produce the final move for this turn.

Required format: {action_format}

Game notes:
{notes}

Code execution log:
{code_log}

Observation:
{observation}

Lessons from earlier turns in this game:
{lessons}

Respond with the action text only (TextArena format). No extra commentary.
"""

VALIDATION_PROMPT = """The previous action failed validation: {error}
Required format: {action_format}
Regenerate the action only.
"""

POST_GAME_PROMPT = """Summarize one lesson learned from this {game_title} trajectory.
One sentence only.

Trajectory summary:
{summary}
"""
