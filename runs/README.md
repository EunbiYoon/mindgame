# Runs

Pipeline outputs (checkpoints, jsonl, metrics) are written under each module's `runs/` folder at runtime and are not tracked in git.

| Module | Output path |
|--------|-------------|
| `mgc2025_sft/runs/` | SFT pipeline metadata |
| `mgc2025_sft/data/runs/` | Converted train/test jsonl |
| `in2ai/runs/` | Rollouts + GRPO checkpoints |
| `stars/runs/` | ReAct trajectories |
| `rlgaming/runs/` | Data + LoRA + metrics |
| `lora/runs/` | Shared LoRA adapters |
| `eval/runs/` | Shared evaluation metrics |

See [README.md](../README.md) for how to run each pipeline.
