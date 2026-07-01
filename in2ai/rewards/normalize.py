"""Environment-specific reward normalization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardNormalizer:
    method: str = "zscore"
    clip: float = 3.0
    _sum: float = 0.0
    _sq_sum: float = 0.0
    _count: int = 0
    _by_game: dict[str, "RewardNormalizer"] = field(default_factory=dict)

    def fit(self, game: str, values: list[float]) -> None:
        norm = self._by_game.setdefault(game, RewardNormalizer(self.method, self.clip))
        for v in values:
            norm._sum += v
            norm._sq_sum += v * v
            norm._count += 1

    def transform(self, game: str, value: float) -> float:
        if self.method == "none":
            return value
        norm = self._by_game.get(game)
        if norm is None or norm._count < 2:
            return value
        mean = norm._sum / norm._count
        var = max(norm._sq_sum / norm._count - mean * mean, 1e-8)
        z = (value - mean) / (var ** 0.5)
        if self.clip:
            z = max(-self.clip, min(self.clip, z))
        return z

    def normalize_episode_steps(self, game: str, steps: list) -> None:
        raw = [s.raw_reward for s in steps if s.eligible]
        self.fit(game, raw)
        for s in steps:
            if s.eligible:
                s.normalized_reward = self.transform(game, s.raw_reward)
