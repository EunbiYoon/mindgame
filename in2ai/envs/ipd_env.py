"""Three-player IPD environment for RL rollouts."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.ipd.engine import (  # noqa: E402
    MAX_ROUNDS,
    PLAYERS,
    format_choice,
    opponent_action,
    opponent_message,
    round_scores,
)
from in2ai.envs.base import EpisodeResult, GameEnv, StepRecord  # noqa: E402
from in2ai.rewards.compute import ipd_episode_bonus, ipd_step_reward  # noqa: E402
from in2ai.validation.eligibility import is_eligible  # noqa: E402
from in2ai.validation.output import validate_completion  # noqa: E402


class IPDEnv(GameEnv):
    game = "ipd"

    def __init__(self, rng: random.Random, reward_cfg: dict[str, Any]):
        self.rng = rng
        self.reward_cfg = reward_cfg
        self.episode_id = ""
        self.self_id = "P0"
        self.strategies: dict[str, str] = {}
        self.round_idx = 1
        self.total_scores = {p: 0 for p in PLAYERS}
        self.last_actions: dict[str, str] = {}
        self.step_idx = 0
        self.caused_error = False
        self._phase = "communication"

    def reset(self, episode_id: str, opponent: str) -> StepRecord:
        self.episode_id = episode_id
        self.self_id = "P0"
        self.strategies = {p: opponent if p != "P0" else "self" for p in PLAYERS}
        self.strategies["P0"] = "self"
        if opponent == "mixed":
            self.strategies = {
                p: self.rng.choice(
                    ["always_cooperate", "always_defect", "tit_for_tat", "grim_trigger", "random"]
                )
                for p in PLAYERS
            }
        self.round_idx = 1
        self.total_scores = {p: 0 for p in PLAYERS}
        self.last_actions = {}
        self.step_idx = 0
        self.caused_error = False
        self._phase = "communication"
        return self._build_step()

    def _game_state(self) -> dict[str, Any]:
        messages = {
            p: (
                "[pending model message]"
                if p == self.self_id and self._phase == "communication"
                else opponent_message(self.strategies[p], p, self.last_actions, self.rng)
            )
            for p in PLAYERS
        }
        return {
            "game": "Three-Player Iterated Prisoner's Dilemma",
            "self_id": self.self_id,
            "players": list(PLAYERS),
            "round": self.round_idx,
            "max_rounds": MAX_ROUNDS,
            "phase": self._phase,
            "scores": dict(self.total_scores),
            "messages": messages,
            "last_actions": dict(self.last_actions),
        }

    def _prompt(self) -> str:
        inp = {"game_state": self._game_state()}
        return (
            "You are playing Three-Player Iterated Prisoner's Dilemma (MindGames).\n"
            "Each round has a communication phase then a simultaneous decision phase.\n"
            "Legal decisions: [cooperate] or [defect]. Payoffs are summed over all pairs.\n\n"
            f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
            "Respond as JSON with keys: belief, reasoning, action."
        )

    def _build_step(self) -> StepRecord:
        return StepRecord(
            game=self.game,
            episode_id=self.episode_id,
            step_idx=self.step_idx,
            phase=self._phase,
            prompt=self._prompt(),
            completion="",
            eligible=is_eligible(self.game, self._phase, True),
            env_state=self.snapshot(),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategies": dict(self.strategies),
            "round_idx": self.round_idx,
            "total_scores": dict(self.total_scores),
            "last_actions": dict(self.last_actions),
            "phase": self._phase,
            "self_id": self.self_id,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.strategies = dict(state["strategies"])
        self.round_idx = state["round_idx"]
        self.total_scores = dict(state["total_scores"])
        self.last_actions = dict(state["last_actions"])
        self._phase = state["phase"]
        self.self_id = state["self_id"]

    def step(self, completion: str) -> tuple[StepRecord | None, bool, dict[str, Any]]:
        state_before = self.snapshot()
        vr = validate_completion(
            self.game, completion, context={"phase": self._phase}
        )
        step = self._build_step()
        step.completion = completion
        step.env_state = state_before
        step.valid_output = vr.valid
        step.invalid_reason = vr.reason

        if self._phase == "communication":
            self._phase = "decision"
            self.step_idx += 1
            next_step = self._build_step()
            next_step.step_idx = self.step_idx
            return next_step, False, {"phase": "communication_done"}

        if not vr.valid:
            self.caused_error = True
            step.raw_reward = ipd_step_reward(-1, **self._step_kw())
            return None, True, {"outcome": "loss", "caused_error": True}

        self_choice = str(vr.parsed)
        opp_actions = {
            p: opponent_action(self.strategies[p], p, self.last_actions, self.rng)
            for p in PLAYERS
            if p != self.self_id
        }
        actions = {self.self_id: self_choice, **opp_actions}
        gained = round_scores(actions)
        step.raw_reward = ipd_step_reward(
            gained[self.self_id], **self._step_kw()
        )
        for p in PLAYERS:
            self.total_scores[p] += gained[p]
        self.last_actions = actions

        self.step_idx += 1
        self.round_idx += 1
        self._phase = "communication"

        if self.round_idx > MAX_ROUNDS:
            best_opp = max(self.total_scores[p] for p in PLAYERS if p != self.self_id)
            bonus = ipd_episode_bonus(
                self.total_scores[self.self_id],
                best_opp,
                win_bonus=self.reward_cfg.get("match_win_bonus", 3.0),
            )
            outcome = (
                "win"
                if self.total_scores[self.self_id] > best_opp
                else "loss"
                if self.total_scores[self.self_id] < best_opp
                else "tie"
            )
            return None, True, {"outcome": outcome, "episode_bonus": bonus}

        return self._build_step(), False, {"round_scores": gained}

    def _step_kw(self) -> dict[str, Any]:
        return {
            "score_scale": self.reward_cfg.get("score_scale", 15.0),
            "invalid": self.reward_cfg.get("invalid_action", -2.0),
        }

    def replay_reward(self, env_state: dict[str, Any], completion: str) -> float:
        self.restore(env_state)
        vr = validate_completion(
            self.game, completion, context={"phase": self._phase}
        )
        if self._phase == "communication":
            return 0.0
        if not vr.valid:
            return ipd_step_reward(-1, **self._step_kw())
        self_choice = str(vr.parsed)
        opp_actions = {
            p: opponent_action(self.strategies[p], p, self.last_actions, self.rng)
            for p in PLAYERS
            if p != self.self_id
        }
        actions = {self.self_id: self_choice, **opp_actions}
        return ipd_step_reward(round_scores(actions)[self.self_id], **self._step_kw())


def run_episode(
    env: IPDEnv,
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
        game="ipd",
        episode_id=episode_id,
        outcome=outcome,
        total_return=total,
        steps=steps,
        caused_error=env.caused_error,
    )
