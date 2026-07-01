#!/usr/bin/env python3
"""GRPO training with In2AI reward shaping (validation + env replay + credit)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_cache import ensure_hf_cache  # noqa: E402
from in2ai.envs.blotto_env import BlottoEnv  # noqa: E402
from in2ai.envs.codenames_env import CodenamesEnv  # noqa: E402
from in2ai.envs.ipd_env import IPDEnv  # noqa: E402
from in2ai.envs.mafia_env import MafiaEnv  # noqa: E402
from in2ai.rewards.compute import game_reward_config  # noqa: E402

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:
    raise SystemExit(
        "trl is required for GRPO training. conda activate maf && pip install trl"
    ) from exc


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_replay_env(game: str, reward_cfg: dict):
    import random

    rng = random.Random(0)
    if game == "blotto":
        return BlottoEnv(rng, reward_cfg)
    if game == "ipd":
        return IPDEnv(rng, reward_cfg)
    if game == "codenames":
        return CodenamesEnv(rng, reward_cfg)
    if game == "mafia":
        return MafiaEnv(rng, reward_cfg)
    raise ValueError(game)


def main():
    ap = argparse.ArgumentParser(description="In2AI GRPO trainer")
    ap.add_argument("--game", required=True, choices=["blotto", "ipd", "codenames", "mafia"])
    ap.add_argument("--dataset_file", required=True, help="prepare_dataset.py output jsonl")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--config", default="in2ai/config/in2ai_default.yaml")
    ap.add_argument("--adapter_path", default=None, help="SFT warm-start LoRA")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    run_id = args.run_id or os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or ROOT / "in2ai" / "runs" / run_id / "checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_hf_cache()
    reward_cfg = game_reward_config(cfg, args.game)
    replay_env = make_replay_env(args.game, reward_cfg)

    ds = load_dataset("json", data_files=args.dataset_file, split="train")
    ds = ds.filter(lambda x: bool(x.get("prompt")))

    # Cache env_state + credit for reward_fn lookups keyed by prompt.
    meta_by_prompt: dict[str, dict] = {}
    for row in ds:
        meta_by_prompt[row["prompt"]] = {
            "env_state": row.get("env_state") or {},
            "credit": float(row.get("credit") or 0.0),
        }

    base_model = cfg["base_model"]
    load_4bit = cfg.get("load_in_4bit", True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        if load_4bit:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        prompt_list = prompts or kwargs.get("prompt") or []
        for prompt, completion in zip(prompt_list, completions):
            meta = meta_by_prompt.get(prompt, {})
            env_state = meta.get("env_state") or {}
            replay = replay_env.replay_reward(env_state, completion)
            # Blend immediate env reward with delayed credit target.
            credit = meta.get("credit", 0.0)
            score = 0.5 * replay + 0.5 * credit
            rewards.append(float(score))
        return rewards

    grpo_cfg = cfg.get("grpo", {})
    training_args = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=grpo_cfg.get("learning_rate", 1e-5),
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 256),
        beta=grpo_cfg.get("beta", 0.04),
        logging_steps=10,
        save_steps=50,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    info = {
        "run_id": run_id,
        "game": args.game,
        "method": "in2ai_grpo",
        "base_model": base_model,
        "dataset_file": args.dataset_file,
        "out_dir": str(out_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved GRPO checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
