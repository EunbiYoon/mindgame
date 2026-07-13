# MindGames Challenge

Reproduction of [MindGames](https://arxiv.org/abs/2505.05522) — four games and the top-three Efficient-division approaches on **Qwen3-8B**.

Paper: [`mindgames.pdf`](mindgames.pdf)

## Repository layout

| Folder | Role |
|--------|------|
| [`games/`](games/) | Game simulators (Mafia, Blotto, IPD, Codenames) |
| [`eunbi/`](eunbi/) | **Shared LoRA train + eval** (`eunbi/lora/`, `eunbi/eval/`) |
| [`mgc2025_sft/`](mgc2025_sft/) | MGC2025 trajectory → LoRA imitation |
| [`in2ai/`](in2ai/) | 1st place — async rollout → GRPO |
| [`stars/`](stars/) | 2nd place — Ollama + ReAct (no fine-tuning) |
| [`rlgaming/`](rlgaming/) | 3rd place — filtered multi-source → LoRA |
| [`scripts/`](scripts/) | Shell helpers, conda hook |
| [`previous_result/`](previous_result/) | Past run artifacts (gitignored) |

## Setup

```bash
cp .env.example .env          # edit MAF_ROOT, cluster paths
module load conda/latest cuda/12.6
conda activate maf
bash scripts/install-conda-hook.sh   # one-time
conda deactivate && conda activate maf
pip install -r requirements.txt
```

Set `RUN_ID` to fix output folder name (default: UTC timestamp).

---

## How to run — 3 modes

Each module supports up to three modes:

| Mode | Meaning |
|------|---------|
| **Train** | Learn / collect training data only |
| **Eval** | Score an existing model (no training) |
| **Train & eval** | Full pipeline in one command |

Replace `<game>` with `mafia`, `blotto`, `ipd`, or `codenames`.

---

### `games/` — simulators

| Mode | Command |
|------|---------|
| Train (generate SFT data) | `python games/<game>/generate_sft.py --out games/<game>/sft.jsonl --n_games 200 --seed 42` |
| Eval | — (use a pipeline below) |
| Train & eval | — |

---

### `eunbi/` — shared LoRA train & eval

Used by `mgc2025_sft` and `rlgaming`. See [`eunbi/README.md`](eunbi/README.md).

| Mode | Command |
|------|---------|
| **Train** | `python eunbi/lora/train_lora.py --train_file <jsonl> --run_id <RUN_ID> --load_in_4bit --epochs 1` |
| **Eval** | `python eunbi/eval/evaluate_<game>.py --model_dir eunbi/lora/runs/<RUN_ID> --test_file <jsonl> --n 100` |
| **Train & eval** | Use `mgc2025_sft/run_<game>.sh` or `rlgaming/run_<game>.sh` |

---

### `mgc2025_sft/` — MGC2025 imitation learning

| Mode | Command |
|------|---------|
| **Train** | `source mgc2025_sft/lib.sh && mgc_setup_root && mgc_new_run_id <game> && maf_train_lora_extra_args && mgc_convert_game <game> && python eunbi/lora/train_lora.py --train_file mgc2025_sft/data/runs/$RUN_ID/<game>_train.jsonl --run_id $RUN_ID ${TRAIN_LORA_EXTRA[@]} --epochs 1` |
| **Eval** | `python mgc2025_sft/evaluate.py --game <game> --model_dir eunbi/lora/runs/<RUN_ID> --test_file mgc2025_sft/data/runs/<RUN_ID>/<game>_test.jsonl --n 100`<br>Blotto 추가: `python eunbi/eval/evaluate_blotto.py --model_dir eunbi/lora/runs/<RUN_ID> --run_id <RUN_ID> --n_games 30` |
| **Train & eval** | `bash mgc2025_sft/run_<game>.sh` |
| All 4 games | `bash mgc2025_sft/run_all.sh` |
| Smoke | `bash mgc2025_sft/run_smoke.sh` |

Data only (convert, no GPU): `MGC_SMOKE_CONVERT_ONLY=1 bash mgc2025_sft/run_smoke.sh`

---

### `in2ai/` — 1st place RL (GRPO)

Rollout + GRPO. No separate `eunbi/eval` step (rewards are used during training).

| Mode | Command |
|------|---------|
| **Train** | `bash in2ai/run_<game>.sh` (rollout → dataset → GRPO) |
| **Eval** | — (no standalone eval script; same as train) |
| **Train & eval** | `bash in2ai/run_<game>.sh` (same as train) |
| All 4 games | `bash in2ai/run_all.sh` |
| Smoke | `RUN_ID=fp_smoke N_EPISODES=4 N_WORKERS=2 bash in2ai/run_blotto.sh` |

SFT warm-start: `FP_ADAPTER=eunbi/lora/runs/<RUN_ID> bash in2ai/run_blotto.sh`

---

### `stars/` — 2nd place STARS (no fine-tuning)

| Mode | Command |
|------|---------|
| Train | — (fine-tuning 없음) |
| **Eval** | `bash stars/run_<game>.sh` |
| Train & eval | — |
| All 4 games | `bash stars/run_all.sh` |
| Dry-run (no Ollama) | `STARS_DRY_RUN=1 STARS_EVAL_N=5 bash stars/run_<game>.sh` |
| Smoke | `bash stars/run_smoke.sh` |

---

### `rlgaming/` — 3rd place filtered SFT

| Mode | Command |
|------|---------|
| **Train** (data + LoRA) | `bash rlgaming/run_data.sh` (data only) |
| **Eval** | `python rlgaming/eval/evaluate.py --game <game> --model_dir rlgaming/runs/<RUN_ID>/lora --test_file rlgaming/runs/<RUN_ID>/<game>_test.jsonl --n 100` |
| **Train & eval** | `bash rlgaming/run_<game>.sh` |
| All 4 games | `bash rlgaming/run_all.sh` |
| Smoke | `bash rlgaming/run_smoke.sh` |

Data only: `RLG_GAME=<game> bash rlgaming/run_data.sh`

---

## Quick reference — one-liners

```bash
# mgc2025_sft — train & eval
bash mgc2025_sft/run_blotto.sh

# in2ai — train (RL)
RUN_ID=fp_smoke N_EPISODES=4 bash in2ai/run_blotto.sh

# stars — eval only
STARS_DRY_RUN=1 bash stars/run_blotto.sh

# rlgaming — train & eval
bash rlgaming/run_blotto.sh
```

## Outputs

| Module | Output path |
|--------|-------------|
| `eunbi/lora/runs/<RUN_ID>/` | LoRA adapter |
| `eunbi/eval/runs/<RUN_ID>/` | metrics json |
| `mgc2025_sft/data/runs/<RUN_ID>/` | train/test jsonl |
| `in2ai/runs/<RUN_ID>/` | rollouts + GRPO checkpoints |
| `stars/runs/<RUN_ID>/` | ReAct trajectories |
| `rlgaming/runs/<RUN_ID>/` | data + lora + metrics |

Artifacts are gitignored under `previous_result/`. See [`previous_result/README.md`](previous_result/README.md).

## Environment (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAF_BASE_MODEL` | `Qwen/Qwen3-8B` | LoRA backbone |
| `MAF_LORA_4BIT` | `1` | 4-bit quantization |
| `MAF_HF_CACHE` | `games/.cache/huggingface` | HF cache |
| `STARS_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API |
