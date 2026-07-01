"""Filter and score RLGaming SFT trajectories."""

from __future__ import annotations

import json
from typing import Any, Iterator


def _player_reward(meta: dict[str, Any]) -> float | None:
    raw = meta.get("rewards")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            rewards = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        rewards = raw
    else:
        return None
    pid = meta.get("player_id")
    if pid is None:
        return None
    val = rewards.get(str(pid), rewards.get(pid))
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _completion_ok(completion: str, *, min_len: int = 1) -> bool:
    text = str(completion or "").strip()
    return len(text) >= min_len


def _is_proprietary_model(model_name: str) -> bool:
    low = str(model_name or "").lower()
    return any(tag in low for tag in ("gpt4", "gpt-4", "gpt5", "gpt-5", "o1", "o3"))


def score_example(ex: dict[str, Any]) -> float:
    meta = ex.get("meta") or {}
    source = meta.get("source", "")
    score = 0.0

    if source == "rule_based":
        score += 1.0
    elif source == "mgc2025":
        score += 0.5
        if _is_proprietary_model(str(meta.get("model_name", ""))):
            score += 1.0
        reward = _player_reward(meta)
        if reward is not None and reward > 0:
            score += 2.0
        if meta.get("status") == "finished":
            score += 0.5
        turns = meta.get("num_turns")
        if isinstance(turns, int) and turns >= 3:
            score += 0.25

    if _completion_ok(ex.get("completion", "")):
        score += 0.5

    prompt = ex.get("prompt", "")
    if "[GAME]" in prompt or "STATE:" in prompt or "OBSERVATION:" in prompt:
        score += 0.25

    return score


def should_keep(
    ex: dict[str, Any],
    *,
    win_only: bool = False,
    min_score: float = 0.0,
    proprietary_only: bool = False,
) -> bool:
    meta = ex.get("meta") or {}
    source = meta.get("source", "")

    if not _completion_ok(ex.get("completion", "")):
        return False

    if proprietary_only and source == "mgc2025":
        if not _is_proprietary_model(str(meta.get("model_name", ""))):
            return False

    if win_only and source == "mgc2025":
        reward = _player_reward(meta)
        if reward is None or reward <= 0:
            return False
        if meta.get("status") != "finished":
            return False

    if score_example(ex) < min_score:
        return False

    return True


def filter_examples(
    examples: Iterator[dict[str, Any]],
    *,
    win_only: bool = False,
    min_score: float = 0.0,
    proprietary_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    stats = {"in": 0, "out": 0, "by_source": {}}
    for ex in examples:
        stats["in"] += 1
        src = (ex.get("meta") or {}).get("source", "unknown")
        if should_keep(ex, win_only=win_only, min_score=min_score, proprietary_only=proprietary_only):
            kept.append(ex)
            stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
        else:
            stats["out"] += 1
    stats["kept"] = len(kept)
    return kept, stats


def dedupe_key(ex: dict[str, Any]) -> str:
    meta = ex.get("meta") or {}
    if meta.get("source") == "mgc2025":
        return f"mgc:{meta.get('player_game_id')}:{meta.get('turn')}"
    inp = ex.get("input") or {}
    priv = inp.get("private_info") or {}
    gs = inp.get("game_state") or {}
    return f"rule:{gs.get('day')}:{priv.get('self_id')}:{hash(ex.get('completion', ''))}"


def dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for ex in examples:
        key = dedupe_key(ex)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out
