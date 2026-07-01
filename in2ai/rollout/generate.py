#!/usr/bin/env python3
"""Collect RL rollouts with In2AI components (validation, eligibility, credit, norm)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.model_utils import generate_text, load_model_and_tokenizer  # noqa: E402
from hf_cache import ensure_hf_cache  # noqa: E402
from in2ai.credit.delayed import assign_credit  # noqa: E402
from in2ai.envs.blotto_env import BlottoEnv, run_episode as run_blotto  # noqa: E402
from in2ai.envs.codenames_env import CodenamesEnv, run_episode as run_codenames  # noqa: E402
from in2ai.envs.ipd_env import IPDEnv, run_episode as run_ipd  # noqa: E402
from in2ai.envs.mafia_env import MafiaEnv, run_episode as run_mafia  # noqa: E402
from in2ai.rewards.compute import game_reward_config  # noqa: E402
from in2ai.rewards.normalize import RewardNormalizer  # noqa: E402
from in2ai.rollout.async_pool import AsyncRolloutPool  # noqa: E402
from in2ai.rollout.opponents import load_curriculum, pick_opponent  # noqa: E402


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_env(game: str, rng: random.Random, cfg: dict):
    reward_cfg = game_reward_config(cfg, game)
    if game == "blotto":
        return BlottoEnv(rng, reward_cfg)
    if game == "ipd":
        return IPDEnv(rng, reward_cfg)
    if game == "codenames":
        return CodenamesEnv(rng, reward_cfg)
    if game == "mafia":
        return MafiaEnv(rng, reward_cfg)
    raise ValueError(f"Unknown game: {game}")


def run_one_episode(
    game: str,
    generate_fn,
    rng: random.Random,
    cfg: dict,
    curriculum: list,
    episode_idx: int,
) -> dict:
    opponent = pick_opponent(game, rng, curriculum, episode_idx)
    env = make_env(game, rng, cfg)
    episode_id = str(uuid.uuid4())[:8]
    runners = {
        "blotto": run_blotto,
        "ipd": run_ipd,
        "codenames": run_codenames,
        "mafia": run_mafia,
    }
    result = runners[game](env, generate_fn, episode_id, opponent)

    assign_credit(
        result.steps,
        episode_bonus=0.0,
        gamma=cfg.get("credit", {}).get("gamma", 0.95),
        method=cfg.get("credit", {}).get("method", "monte_carlo"),
    )
    # Add episode bonus to last eligible step credit.
    bonus = 0.0
    if result.steps:
        eligible = [s for s in result.steps if s.eligible]
        if eligible:
            # episode bonus applied in env step info — recompute from outcome
            from in2ai.rewards.compute import (
                blotto_episode_bonus,
                ipd_episode_bonus,
                mafia_episode_bonus,
            )

            reward_cfg = game_reward_config(cfg, game)
            if game == "blotto":
                bonus = blotto_episode_bonus(
                    result.outcome,
                    match_win=reward_cfg.get("match_win", 5.0),
                    match_loss=reward_cfg.get("match_loss", -5.0),
                )
            elif game == "ipd" and result.outcome != "tie":
                bonus = reward_cfg.get("match_win_bonus", 3.0) * (
                    1 if result.outcome == "win" else -1
                )
            elif game == "codenames" and result.outcome != "tie":
                bonus = 3.0 if result.outcome == "win" else -3.0
            elif game == "mafia" and result.outcome != "tie":
                bonus = mafia_episode_bonus(
                    result.outcome,
                    match_win=reward_cfg.get("match_win", 5.0),
                    match_loss=reward_cfg.get("match_loss", -5.0),
                )
            eligible[-1].credit += bonus

    norm = RewardNormalizer(
        method=cfg.get("reward_norm", {}).get("method", "zscore"),
        clip=cfg.get("reward_norm", {}).get("clip", 3.0),
    )
    norm.normalize_episode_steps(game, result.steps)

    payload = result.to_dict()
    payload["opponent"] = opponent
    payload["curriculum_episode"] = episode_idx
    return payload


def main():
    ap = argparse.ArgumentParser(description="In2AI async rollout collection")
    ap.add_argument("--game", required=True, choices=["blotto", "ipd", "codenames", "mafia"])
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--config", default="in2ai/config/in2ai_default.yaml")
    ap.add_argument("--curriculum", default="in2ai/config/curriculum.yaml")
    ap.add_argument("--model_dir", default=None, help="LoRA dir or base model path")
    ap.add_argument("--n_episodes", type=int, default=None)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    curriculum = load_curriculum(ROOT / args.curriculum)
    run_id = args.run_id or os.environ.get("RUN_ID") or "rl_run"
    out_dir = ROOT / "in2ai" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = args.n_episodes or cfg["rollout"]["n_episodes"]
    n_workers = args.n_workers or cfg["rollout"]["n_workers"]
    model_path = args.model_dir or cfg.get("adapter_path") or cfg["base_model"]

    ensure_hf_cache()
    load_4bit = cfg.get("load_in_4bit", True)
    if "MAF_LORA_4BIT" in os.environ:
        load_4bit = os.environ["MAF_LORA_4BIT"] == "1"
    model, tokenizer, _ = load_model_and_tokenizer(model_path, load_in_4bit=load_4bit)
    max_new = cfg["rollout"]["max_new_tokens"]

    def generate_fn(prompt: str) -> str:
        return generate_text(model, tokenizer, prompt, max_new_tokens=max_new)

    pool = AsyncRolloutPool(n_workers=n_workers)
    gen = pool.wrap_generator(generate_fn)
    rng = random.Random(args.seed)

    tasks = []
    for i in range(n_episodes):
        tasks.append(
            (
                run_one_episode,
                (args.game, gen, random.Random(rng.randint(0, 2**31 - 1)), cfg, curriculum, i),
                {},
            )
        )

    episodes = pool.run(tasks)

    rollout_path = out_dir / f"{args.game}_rollouts.jsonl"
    with rollout_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            for step in ep["steps"]:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "game": args.game,
        "n_episodes": n_episodes,
        "n_workers": n_workers,
        "model": model_path,
        "rollout_file": str(rollout_path.relative_to(ROOT)),
        "wins": sum(1 for e in episodes if e["outcome"] == "win"),
        "losses": sum(1 for e in episodes if e["outcome"] == "loss"),
        "errors": sum(1 for e in episodes if e["caused_error"]),
    }
    (out_dir / f"{args.game}_rollout_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
