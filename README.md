# MindGames Challenge

[MindGames](https://arxiv.org/abs/2505.05522) reproduction — 4 games, top-3 approaches, **Qwen3-8B**, 5 different configurations.

| Folder | What |
|--------|------|
| `eunbi/` | Shared LoRA train + eval |
| `mgc2025_sft/` | MGC2025 → LoRA |
| `in2ai/` | 1st — GRPO |
| `stars/` | 2nd — Ollama ReAct |
| `rlgaming/` | 3rd — filtered SFT |

## Setup

```bash
cp .env.example .env && conda activate maf
bash scripts/install-conda-hook.sh   # one-time
pip install -r requirements.txt
```

Replace `blotto` with `mafia` / `ipd` / `codenames`. Set run name: `RUN_ID=my_run bash ...`

---

## Command

### `mgc2025_sft/`

| | Command |
|---|---------|
| **train** | `MGC_SMOKE_CONVERT_ONLY=1 MGC_SMOKE_GAME=blotto bash mgc2025_sft/run_smoke.sh` (data) then `python eunbi/lora/train_lora.py --train_file mgc2025_sft/data/runs/$RUN_ID/blotto_train.jsonl --run_id $RUN_ID --load_in_4bit --epochs 1` |
| **eval** | `python mgc2025_sft/evaluate.py --game blotto --model_dir eunbi/lora/runs/$RUN_ID --test_file mgc2025_sft/data/runs/$RUN_ID/blotto_test.jsonl --n 100` |
| **train+eval** | `bash mgc2025_sft/run_blotto.sh` |
| all games | `bash mgc2025_sft/run_all.sh` |
| smoke | `bash mgc2025_sft/run_smoke.sh` |

### `in2ai/`

| | Command |
|---|---------|
| **train** | `bash in2ai/run_blotto.sh` |
| **eval** | — |
| **train+eval** | `bash in2ai/run_blotto.sh` |
| all games | `bash in2ai/run_all.sh` |
| smoke | `RUN_ID=fp_smoke N_EPISODES=4 bash in2ai/run_blotto.sh` |

### `stars/`

| | Command |
|---|---------|
| **train** | — |
| **eval** | `bash stars/run_blotto.sh` |
| **train+eval** | — |
| all games | `bash stars/run_all.sh` |
| dry-run | `STARS_DRY_RUN=1 bash stars/run_blotto.sh` |

### `rlgaming/`

| | Command |
|---|---------|
| **train** | `RLG_GAME=blotto bash rlgaming/run_data.sh` (data only) |
| **eval** | `python rlgaming/eval/evaluate.py --game blotto --model_dir rlgaming/runs/$RUN_ID/lora --test_file rlgaming/runs/$RUN_ID/blotto_test.jsonl --n 100` |
| **train+eval** | `bash rlgaming/run_blotto.sh` |
| all games | `bash rlgaming/run_all.sh` |
| smoke | `bash rlgaming/run_smoke.sh` |

### `eunbi/` (Implement with Lora)

**LoRA:** `r=16`, `alpha=32`, `dropout=0.05`, 4-bit, `Qwen3-8B`, targets all 7 linear layers (`q/k/v/o_proj`, `gate/up/down_proj`). Same in `in2ai` GRPO.

| | Command |
|---|---------|
| **train** | `python eunbi/lora/train_lora.py --train_file <jsonl> --run_id $RUN_ID --load_in_4bit --epochs 1` |
| **eval** | `python eunbi/eval/evaluate_blotto.py --model_dir eunbi/lora/runs/$RUN_ID --run_id $RUN_ID --n_games 30` |
| **train+eval** | `bash eunbi/run_blotto.sh` |
| all games | `bash eunbi/run_all.sh` |
| smoke | `bash eunbi/run_smoke.sh` |


---

## Outputs

`eunbi/lora/runs/<RUN_ID>/` · `eunbi/eval/runs/<RUN_ID>/` · `eunbi/data/runs/<RUN_ID>/` · `mgc2025_sft/data/runs/<RUN_ID>/` · `in2ai/runs/<RUN_ID>/` · `stars/runs/<RUN_ID>/` · `rlgaming/runs/<RUN_ID>/`

## Env (`.env`)

`MAF_BASE_MODEL=Qwen/Qwen3-8B` · `MAF_LORA_4BIT=1` · `STARS_OLLAMA_HOST=http://127.0.0.1:11434`
