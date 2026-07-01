"""Codenames environment (simplified operative turns for RL)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.codenames.engine import (  # noqa: E402
    new_board,
    parse_guess,
    public_board,
    team_words,
    unrevealed_words,
)
from in2ai.envs.base import EpisodeResult, GameEnv, StepRecord  # noqa: E402
from in2ai.rewards.compute import codenames_step_reward  # noqa: E402
from in2ai.validation.eligibility import is_eligible  # noqa: E402
from in2ai.validation.output import validate_completion  # noqa: E402


class CodenamesEnv(GameEnv):
    game = "codenames"

    def __init__(self, rng: random.Random, reward_cfg: dict[str, Any], skill: str = "random"):
        self.rng = rng
        self.reward_cfg = reward_cfg
        self.skill = skill
        self.episode_id = ""
        self.board: dict = {}
        self.revealed: set[str] = set()
        self.team = "red"
        self.step_idx = 0
        self.caused_error = False
        self.max_turns = 12

    def reset(self, episode_id: str, opponent: str) -> StepRecord:
        self.episode_id = episode_id
        self.skill = opponent or self.skill
        self.board = new_board(self.rng, "red")
        self.revealed = set()
        self.team = "red"
        self.step_idx = 0
        self.caused_error = False
        return self._build_step()

    def _build_step(self) -> StepRecord:
        board_words = [w.lower() for w in self.board["words"]]
        inp = {
            "game": "Codenames",
            "role": "operative",
            "team": self.team,
            "board": public_board(self.board, self.revealed),
            "remaining_team_words": len(
                [w for w in team_words(self.board, self.team) if w not in self.revealed]
            ),
        }
        prompt = (
            "You are the red operative in Codenames (MindGames).\n"
            "Guess with [word] or [pass].\n\n"
            f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
            "Respond as JSON with keys: belief, reasoning, action."
        )
        return StepRecord(
            game=self.game,
            episode_id=self.episode_id,
            step_idx=self.step_idx,
            phase="operative",
            prompt=prompt,
            completion="",
            eligible=is_eligible(self.game, "operative", True),
            env_state=self.snapshot(),
            meta={"board_words": board_words},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "revealed": list(self.revealed),
            "team": self.team,
            "step_idx": self.step_idx,
            "skill": self.skill,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.board = state["board"]
        self.revealed = set(state["revealed"])
        self.team = state["team"]
        self.step_idx = state["step_idx"]
        self.skill = state["skill"]

    def _resolve_guess(self, guess: str) -> tuple[str, bool]:
        if guess == "pass":
            return "pass", False
        label = self.board["labels"].get(guess, "neutral")
        self.revealed.add(guess)
        if label == "assassin":
            return "assassin", True
        if label == self.team:
            return "team_word", False
        if label in ("red", "blue") and label != self.team:
            return "wrong_team", True
        return "neutral", True

    def step(self, completion: str) -> tuple[StepRecord | None, bool, dict[str, Any]]:
        state_before = self.snapshot()
        board_words = [w.lower() for w in self.board["words"]]
        vr = validate_completion(
            self.game,
            completion,
            context={"role": "operative", "board_words": board_words},
        )
        step = self._build_step()
        step.completion = completion
        step.env_state = state_before
        step.valid_output = vr.valid
        step.invalid_reason = vr.reason

        if not vr.valid:
            self.caused_error = True
            step.raw_reward = codenames_step_reward("invalid", **self._step_kw())
            return None, True, {"outcome": "loss", "caused_error": True}

        guess = str(vr.parsed) if vr.parsed != "pass" else "pass"
        if guess != "pass":
            guess = guess.lower()
        event, terminal = self._resolve_guess(guess)
        step.raw_reward = codenames_step_reward(event, **self._step_kw())

        self.step_idx += 1
        remaining = [
            w for w in team_words(self.board, self.team) if w not in self.revealed
        ]

        if terminal or not remaining or self.step_idx >= self.max_turns:
            outcome = "win" if remaining == [] and event == "team_word" else "loss"
            if event == "assassin":
                outcome = "loss"
            bonus = 3.0 if outcome == "win" else -3.0
            return None, True, {"outcome": outcome, "episode_bonus": bonus}

        return self._build_step(), False, {"event": event}

    def _step_kw(self) -> dict[str, Any]:
        return {
            "team_word": self.reward_cfg.get("team_word", 1.0),
            "wrong_team": self.reward_cfg.get("wrong_team", -1.0),
            "neutral": self.reward_cfg.get("neutral", -0.5),
            "assassin": self.reward_cfg.get("assassin", -5.0),
            "invalid": self.reward_cfg.get("invalid_action", -2.0),
        }

    def replay_reward(self, env_state: dict[str, Any], completion: str) -> float:
        self.restore(env_state)
        board_words = [w.lower() for w in self.board["words"]]
        vr = validate_completion(
            self.game,
            completion,
            context={"role": "operative", "board_words": board_words},
        )
        if not vr.valid:
            return codenames_step_reward("invalid", **self._step_kw())
        guess = str(vr.parsed) if vr.parsed != "pass" else "pass"
        if guess != "pass":
            guess = guess.lower()
        event, _ = self._resolve_guess(guess)
        return codenames_step_reward(event, **self._step_kw())


def run_episode(
    env: CodenamesEnv,
    generate_fn,
    episode_id: str,
    opponent: str,
) -> EpisodeResult:
    step = env.reset(episode_id, opponent)
    steps: list[StepRecord] = []
    episode_bonus = 0.0
    outcome = "tie"

    while step is not None:
        completion = generate_fn(step.prompt)
        step.completion = completion
        steps.append(step)
        next_step, done, info = env.step(completion)
        if done:
            outcome = info.get("outcome", "tie")
            episode_bonus = info.get("episode_bonus", 0.0)
            break
        step = next_step

    total = sum(s.raw_reward for s in steps if s.eligible) + episode_bonus
    return EpisodeResult(
        game="codenames",
        episode_id=episode_id,
        outcome=outcome,
        total_return=total,
        steps=steps,
        caused_error=env.caused_error,
    )
