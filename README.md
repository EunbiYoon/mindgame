# MindGames Challenge

Reproduction of [MindGames](https://arxiv.org/abs/2505.05522) — four games and the top-three Efficient-division approaches on **Qwen3-8B**.

Paper: [`mindgames.pdf`](mindgames.pdf)

## Five modules

| Folder | Approach | Key idea |
|--------|----------|----------|
| [`games/`](games/) | Game engines | Rule-based simulators for Mafia, Blotto, IPD, Codenames |
| [`mgc2025_sft/`](mgc2025_sft/) | MGC2025 imitation | Competition trajectories → LoRA SFT |
| [`in2ai/`](in2ai/) | 1st place (In2AI) | Async rollout → delayed credit → GRPO |
| [`stars/`](stars/) | 2nd place (STARS) | Ollama + ReAct + PAL (no fine-tuning) |
| [`rlgaming/`](rlgaming/) | 3rd place (RLGaming) | Filtered multi-source trajectories → LoRA |

Shared training/eval: `lora/`, `eval/`, `scripts/`

## Setup

```bash
cp .env.example .env
module load conda/latest cuda/12.6
conda activate maf
bash scripts/install-conda-hook.sh   # one-time
conda deactivate && conda activate maf
pip install -r requirements.txt
```

## Quick start

```bash
bash mgc2025_sft/run_smoke.sh
RUN_ID=fp_smoke N_EPISODES=4 bash in2ai/run_blotto.sh
STARS_DRY_RUN=1 bash stars/run_blotto.sh
bash rlgaming/run_smoke.sh
```

Run all four games: `bash <module>/run_all.sh`

## Module layout

### `games/` — simulators

```text
games/
  mafia/      engine.py  generate_sft.py  schema.json
  blotto/
  ipd/
  codenames/
```

### `mgc2025_sft/` — trajectory imitation

```text
mgc2025_sft/
  convert.py  evaluate.py  lib.sh
  run_{game}.sh  run_all.sh  run_smoke.sh
  data/         # HF cache + converted jsonl (gitignored)
  runs/         # pipeline metadata (gitignored)
```

### `in2ai/` — RL + GRPO

```text
in2ai/
  config/       # model + curriculum yaml
  credit/       # delayed credit assignment
  envs/         # RL game environments
  rewards/      # per-game reward + normalization
  rollout/      # async self-play data collection
  train/        # dataset prep + GRPO
  validation/   # output check + eligibility gating
  run_*.sh
  runs/
```

### `stars/` — ReAct agent (no FT)

```text
stars/
  agent/        # ReAct loop, belief state, code execution
  prompts/      # questionnaire + game prompts
  parse/        # observation parsing + schemas
  games.py      # per-game scoring
  ollama_client.py
  evaluate_game.py
  run_*.sh
  runs/
```

### `rlgaming/` — filtered multi-source SFT

```text
rlgaming/
  data/         # rule agents, MGC merge, trajectory filter
  format/       # [GAME] / [PRIVATE] context blocks
  eval/         # LoRA evaluation + scoring
  run_*.sh
  runs/
```

## Outputs

Runtime artifacts go to `*/runs/<RUN_ID>/` and are gitignored. See [`runs/README.md`](runs/README.md).

```text
mgc2025_sft/data/runs/<RUN_ID>/   # train/test jsonl
lora/runs/<RUN_ID>/               # LoRA adapter
eval/runs/<RUN_ID>/               # metrics
in2ai/runs/<RUN_ID>/              # rollouts + GRPO checkpoints
stars/runs/<RUN_ID>/              # ReAct trajectories
rlgaming/runs/<RUN_ID>/           # data + LoRA + metrics
```

## Environment (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAF_BASE_MODEL` | `Qwen/Qwen3-8B` | LoRA backbone |
| `MAF_LORA_4BIT` | `1` | 4-bit quantization |
| `MAF_HF_CACHE` | `games/.cache/huggingface` | HF cache |
| `STARS_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API |
