"""Game environment base types for In2AI-style RL rollouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    game: str
    episode_id: str
    step_idx: int
    phase: str
    prompt: str
    completion: str
    raw_reward: float = 0.0
    normalized_reward: float = 0.0
    credit: float = 0.0
    eligible: bool = True
    valid_output: bool = False
    invalid_reason: str = ""
    done: bool = False
    env_state: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    game: str
    episode_id: str
    outcome: str
    total_return: float
    steps: list[StepRecord] = field(default_factory=list)
    caused_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "total_return": self.total_return,
            "caused_error": self.caused_error,
            "n_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


class GameEnv(ABC):
    game: str

    @abstractmethod
    def reset(self, episode_id: str, opponent: str) -> StepRecord | None:
        """Start episode; return first decision step or None if immediate terminal."""

    @abstractmethod
    def step(self, completion: str) -> tuple[StepRecord | None, bool, dict[str, Any]]:
        """Apply model action; return (next_step, done, info)."""

    @abstractmethod
    def replay_reward(self, env_state: dict[str, Any], completion: str) -> float:
        """Score a counterfactual completion from a saved env_state (GRPO reward_fn)."""
