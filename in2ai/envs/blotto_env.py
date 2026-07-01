"""Colonel Blotto environment for RL rollouts."""

from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.blotto_match import (  # noqa: E402
    MAX_ROUNDS,
    UNITS,
    WIN_ROUNDS,
    build_blotto_prompt,
    opponent_allocate,
    parse_model_allocation,
    resolve_round,
)
from in2ai.envs.base import EpisodeResult, GameEnv, StepRecord  # noqa: E402
from in2ai.rewards.compute import blotto_episode_bonus, blotto_step_reward, game_reward_config  # noqa: E402
from in2ai.validation.eligibility import is_eligible  # noqa: E402
from in2ai.validation.output import validate_completion  # noqa: E402


class BlottoEnv(GameEnv):
    game = "blotto"

    def __init__(self, rng: random.Random, reward_cfg: dict[str, Any]):
        self.rng = rng
        self.reward_cfg = reward_cfg
        self.episode_id = ""
        self.opponent = "balanced"
        self.round_idx = 1
        self.self_round_wins = 0
        self.opp_round_wins = 0
        self.round_history: list[dict] = []
        self.opp_history: list[dict] = []
        self.step_idx = 0
        self.caused_error = False

    def reset(self, episode_id: str, opponent: str) -> StepRecord:
        self.episode_id = episode_id
        self.opponent = opponent
        self.round_idx = 1
        self.self_round_wins = 0
        self.opp_round_wins = 0
        self.round_history = []
        self.opp_history = []
        self.step_idx = 0
        self.caused_error = False
        return self._current_step()

    def _current_step(self) -> StepRecord:
        prompt = build_blotto_prompt(
            self.round_idx, self.self_round_wins, self.opp_round_wins, self.round_history
        )
        return StepRecord(
            game=self.game,
            episode_id=self.episode_id,
            step_idx=self.step_idx,
            phase="allocation",
            prompt=prompt,
            completion="",
            eligible=is_eligible(self.game, "allocation", True),
            env_state=self.snapshot(),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "round_idx": self.round_idx,
            "self_round_wins": self.self_round_wins,
            "opp_round_wins": self.opp_round_wins,
            "round_history": list(self.round_history),
            "opp_history": [dict(x) for x in self.opp_history],
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.opponent = state["opponent"]
        self.round_idx = state["round_idx"]
        self.self_round_wins = state["self_round_wins"]
        self.opp_round_wins = state["opp_round_wins"]
        self.round_history = list(state["round_history"])
        self.opp_history = [dict(x) for x in state["opp_history"]]

    def step(self, completion: str) -> tuple[StepRecord | None, bool, dict[str, Any]]:
        state_before = self.snapshot()
        vr = validate_completion(self.game, completion)
        step = self._current_step()
        step.completion = completion
        step.env_state = state_before
        step.valid_output = vr.valid
        step.invalid_reason = vr.reason

        if self.self_round_wins >= WIN_ROUNDS or self.opp_round_wins >= WIN_ROUNDS:
            return None, True, {"outcome": "terminal"}

        if not vr.valid:
            self.caused_error = True
            step.raw_reward = blotto_step_reward("invalid", **self._step_kw())
            return None, True, {"outcome": "loss", "caused_error": True}

        alloc = parse_model_allocation(completion)
        opp_alloc = opponent_allocate(
            self.opponent, self.rng, self.round_idx - 1, self.opp_history
        )
        outcome = resolve_round(alloc, opp_alloc)
        step.raw_reward = blotto_step_reward(outcome, **self._step_kw())

        from eval.blotto_match import format_allocation  # noqa: E402

        self.round_history.append(
            {
                "round": self.round_idx,
                "self_allocation": format_allocation(alloc),
                "opponent_allocation": format_allocation(opp_alloc),
                "round_winner": outcome,
            }
        )
        self.opp_history.append(opp_alloc)
        if outcome == "self":
            self.self_round_wins += 1
        elif outcome == "opponent":
            self.opp_round_wins += 1

        self.step_idx += 1
        self.round_idx += 1

        if self.self_round_wins >= WIN_ROUNDS or self.opp_round_wins >= WIN_ROUNDS:
            match = "win" if self.self_round_wins > self.opp_round_wins else "loss"
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        if self.round_idx > MAX_ROUNDS:
            if self.self_round_wins > self.opp_round_wins:
                match = "win"
            elif self.opp_round_wins > self.self_round_wins:
                match = "loss"
            else:
                match = "tie"
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        return self._current_step(), False, {"round_outcome": outcome}

    def _step_kw(self) -> dict[str, Any]:
        return {
            "round_win": self.reward_cfg.get("round_win", 1.0),
            "round_loss": self.reward_cfg.get("round_loss", -1.0),
            "invalid": self.reward_cfg.get("invalid_action", -2.0),
        }

    def _match_bonus(self, match: str) -> float:
        if match == "tie":
            return 0.0
        return blotto_episode_bonus(
            match,
            match_win=self.reward_cfg.get("match_win", 5.0),
            match_loss=self.reward_cfg.get("match_loss", -5.0),
        )

    def replay_reward(self, env_state: dict[str, Any], completion: str) -> float:
        self.restore(env_state)
        vr = validate_completion(self.game, completion)
        if not vr.valid:
            return blotto_step_reward("invalid", **self._step_kw())
        alloc = parse_model_allocation(completion)
        opp_alloc = opponent_allocate(
            self.opponent, self.rng, self.round_idx - 1, self.opp_history
        )
        outcome = resolve_round(alloc, opp_alloc)
        return blotto_step_reward(outcome, **self._step_kw())


def run_episode(
    env: BlottoEnv,
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

    total = sum(s.raw_reward for s in steps) + episode_bonus
    return EpisodeResult(
        game="blotto",
        episode_id=episode_id,
        outcome=outcome,
        total_return=total,
        steps=steps,
        caused_error=env.caused_error,
    )
