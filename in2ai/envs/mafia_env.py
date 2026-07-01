"""Secret Mafia environment for In2AI RL rollouts."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.mafia.engine import (  # noqa: E402
    MAX_DAYS,
    PLAYERS,
    assign_roles,
    build_prompt_input,
    check_winner,
    format_prompt,
    mafia_player,
    night_kill,
    opponent_speak,
    opponent_vote,
    tally_votes,
    vote_outcome,
)
from in2ai.envs.base import EpisodeResult, GameEnv, StepRecord  # noqa: E402
from in2ai.rewards.compute import mafia_episode_bonus, mafia_step_reward  # noqa: E402
from in2ai.validation.eligibility import is_eligible  # noqa: E402
from in2ai.validation.output import validate_completion  # noqa: E402


class MafiaEnv(GameEnv):
    game = "mafia"

    def __init__(self, rng: random.Random, reward_cfg: dict[str, Any]):
        self.rng = rng
        self.reward_cfg = reward_cfg
        self.episode_id = ""
        self.opponent = "heuristic"
        self.self_id = "A"
        self.roles: dict[str, str] = {}
        self.alive: list[str] = []
        self.dead: list[str] = []
        self.day = 1
        self.public_chat: list[dict[str, str]] = []
        self.vote_history: list[dict[str, Any]] = []
        self.claims: dict[str, str] = {}
        self.step_idx = 0
        self.caused_error = False
        self._pending_eliminated: str | None = None

    def reset(self, episode_id: str, opponent: str) -> StepRecord:
        self.episode_id = episode_id
        self.opponent = opponent or "heuristic"
        self.self_id = "A"
        self.roles = assign_roles(self.rng)
        self.alive = PLAYERS[:]
        self.dead = []
        self.day = 1
        self.public_chat = []
        self.vote_history = []
        self.claims = {}
        self.step_idx = 0
        self.caused_error = False
        self._pending_eliminated = None
        self._seed_discussion()
        return self._current_step()

    def _seed_discussion(self) -> None:
        for player in self.alive:
            if player == self.self_id:
                continue
            target = opponent_vote(self.opponent, player, self.roles, self.alive, self.rng)
            text = opponent_speak(self.opponent, player, self.roles, target, self.rng)
            self.public_chat.append({"speaker": player, "text": text})
            if self.roles[player] == "Detective" and self.rng.random() < 0.7:
                self.claims[player] = "Detective"
            elif self.roles[player] == "Mafia" and self.rng.random() < 0.25:
                self.claims[player] = self.rng.choice(["Citizen", "Detective"])
            else:
                self.claims[player] = "Citizen"

    def _valid_targets(self) -> list[str]:
        return [p for p in self.alive if p != self.self_id]

    def _current_step(self) -> StepRecord:
        inp = build_prompt_input(
            day=self.day,
            alive=list(self.alive),
            dead=list(self.dead),
            public_chat=list(self.public_chat),
            vote_history=list(self.vote_history),
            claims=dict(self.claims),
            self_id=self.self_id,
            self_role=self.roles[self.self_id],
        )
        prompt = format_prompt(inp)
        return StepRecord(
            game=self.game,
            episode_id=self.episode_id,
            step_idx=self.step_idx,
            phase="vote",
            prompt=prompt,
            completion="",
            eligible=is_eligible(self.game, "vote", True),
            env_state=self.snapshot(),
            meta={"valid_targets": self._valid_targets()},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "self_id": self.self_id,
            "roles": dict(self.roles),
            "alive": list(self.alive),
            "dead": list(self.dead),
            "day": self.day,
            "public_chat": list(self.public_chat),
            "vote_history": [dict(v) for v in self.vote_history],
            "claims": dict(self.claims),
            "step_idx": self.step_idx,
            "pending_eliminated": self._pending_eliminated,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.opponent = state["opponent"]
        self.self_id = state["self_id"]
        self.roles = dict(state["roles"])
        self.alive = list(state["alive"])
        self.dead = list(state["dead"])
        self.day = state["day"]
        self.public_chat = list(state["public_chat"])
        self.vote_history = [dict(v) for v in state["vote_history"]]
        self.claims = dict(state["claims"])
        self.step_idx = state["step_idx"]
        self._pending_eliminated = state.get("pending_eliminated")

    def _step_kw(self) -> dict[str, Any]:
        return {
            "correct_vote": self.reward_cfg.get("correct_vote", 1.0),
            "wrong_vote": self.reward_cfg.get("wrong_vote", -1.0),
            "neutral_vote": self.reward_cfg.get("neutral_vote", 0.0),
            "invalid": self.reward_cfg.get("invalid_action", -2.0),
        }

    def _match_bonus(self, match: str) -> float:
        if match == "tie":
            return 0.0
        return mafia_episode_bonus(
            match,
            match_win=self.reward_cfg.get("match_win", 5.0),
            match_loss=self.reward_cfg.get("match_loss", -5.0),
        )

    def _team_outcome(self, winner: str) -> str:
        self_is_mafia = self.roles[self.self_id] == "Mafia"
        if winner == "mafia":
            return "win" if self_is_mafia else "loss"
        return "loss" if self_is_mafia else "win"

    def _resolve_day(self, votes: dict[str, str]) -> tuple[str | None, dict[str, Any]]:
        eliminated = tally_votes(votes, self.rng)
        self.vote_history.append({"day": self.day, "votes": dict(votes)})
        if eliminated and eliminated in self.alive:
            self.alive.remove(eliminated)
            self.dead.append(eliminated)
        return eliminated, {"eliminated": eliminated, "votes": votes}

    def _resolve_night(self) -> str | None:
        doctor = next((p for p in self.alive if self.roles[p] == "Doctor"), None)
        protect = None
        if doctor and doctor != self.self_id:
            protect = opponent_vote(self.opponent, doctor, self.roles, self.alive, self.rng)
        victim = night_kill(self.roles, self.alive, self.rng, doctor_protect=protect)
        if victim and victim in self.alive:
            self.alive.remove(victim)
            self.dead.append(victim)
            self.public_chat.append({"speaker": "SYSTEM", "text": f"Night falls. {victim} was eliminated."})
        return victim

    def _advance_day(self) -> None:
        self.day += 1
        self.public_chat = []
        self._seed_discussion()

    def step(self, completion: str) -> tuple[StepRecord | None, bool, dict[str, Any]]:
        state_before = self.snapshot()
        valid_targets = self._valid_targets()
        vr = validate_completion(
            self.game,
            completion,
            context={"valid_targets": valid_targets},
        )
        step = self._current_step()
        step.completion = completion
        step.env_state = state_before
        step.valid_output = vr.valid
        step.invalid_reason = vr.reason

        winner = check_winner(self.roles, self.alive)
        if winner:
            match = self._team_outcome(winner)
            step.raw_reward = mafia_step_reward("neutral", **self._step_kw())
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        if not vr.valid:
            self.caused_error = True
            step.raw_reward = mafia_step_reward("invalid", **self._step_kw())
            return None, True, {"outcome": "loss", "caused_error": True}

        speak, vote = vr.parsed  # type: ignore[misc]
        self.public_chat.append({"speaker": self.self_id, "text": speak or f"I vote {vote}."})

        votes = {self.self_id: vote}
        for player in self.alive:
            if player == self.self_id:
                continue
            votes[player] = opponent_vote(self.opponent, player, self.roles, self.alive, self.rng)

        eliminated, _ = self._resolve_day(votes)
        outcome = vote_outcome(vote, eliminated, self.roles, self.self_id)
        step.raw_reward = mafia_step_reward(outcome, **self._step_kw())

        winner = check_winner(self.roles, self.alive)
        if winner:
            match = self._team_outcome(winner)
            self.step_idx += 1
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        if self.day >= MAX_DAYS:
            mafia_alive = sum(1 for p in self.alive if self.roles[p] == "Mafia")
            match = "win" if mafia_alive == 0 else ("loss" if mafia_alive >= len(self.alive) - mafia_alive else "tie")
            if match == "tie":
                match = "loss" if self.roles[self.self_id] != "Mafia" else "win"
            self.step_idx += 1
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        self._resolve_night()
        winner = check_winner(self.roles, self.alive)
        if winner:
            match = self._team_outcome(winner)
            self.step_idx += 1
            return None, True, {"outcome": match, "episode_bonus": self._match_bonus(match)}

        self._advance_day()
        self.step_idx += 1
        return self._current_step(), False, {"vote_outcome": outcome, "eliminated": eliminated}

    def replay_reward(self, env_state: dict[str, Any], completion: str) -> float:
        self.restore(env_state)
        valid_targets = self._valid_targets()
        vr = validate_completion(
            self.game,
            completion,
            context={"valid_targets": valid_targets},
        )
        if not vr.valid:
            return mafia_step_reward("invalid", **self._step_kw())
        _, vote = vr.parsed  # type: ignore[misc]
        votes = {self.self_id: vote}
        for player in self.alive:
            if player == self.self_id:
                continue
            votes[player] = opponent_vote(self.opponent, player, self.roles, self.alive, self.rng)
        eliminated = tally_votes(votes, self.rng)
        outcome = vote_outcome(vote, eliminated, self.roles, self.self_id)
        return mafia_step_reward(outcome, **self._step_kw())


def run_episode(
    env: MafiaEnv,
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
        game="mafia",
        episode_id=episode_id,
        outcome=outcome,
        total_return=total,
        steps=steps,
        caused_error=env.caused_error,
    )
