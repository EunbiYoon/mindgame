# eunbi — shared LoRA train & eval

Combines the former root-level `lora/` and `eval/` into one module.  
Used by `mgc2025_sft` and `rlgaming`.

```text
eunbi/
├── lib.sh
├── run_game.sh            # train + eval (one game)
├── run_{game}.sh          # blotto / mafia / ipd / codenames
├── run_all.sh
├── run_smoke.sh
├── data/runs/<RUN_ID>/    # auto-generated train/test jsonl (gitignored)
├── lora/
│   ├── train_lora.py      # LoRA SFT training
│   └── runs/<RUN_ID>/     # adapters (gitignored)
└── eval/
    ├── evaluate_mafia.py
    ├── evaluate_blotto.py
    ├── evaluate_ipd.py
    ├── evaluate_codenames.py
    ├── metrics.py
    ├── model_utils.py
    └── runs/<RUN_ID>/     # metrics (gitignored)
```

## Train only

```bash
python eunbi/lora/train_lora.py \
  --train_file mgc2025_sft/data/runs/<RUN_ID>/blotto_train.jsonl \
  --run_id <RUN_ID> \
  --load_in_4bit \
  --epochs 1
```

## Eval only

```bash
python eunbi/eval/evaluate_mafia.py \
  --model_dir eunbi/lora/runs/<RUN_ID> \
  --test_file mgc2025_sft/data/runs/<RUN_ID>/mafia_test.jsonl \
  --n 100
```

Omit `--model_dir` to use the latest run from `eunbi/lora/latest.json`.

## Train & eval (one command)

```bash
# 데이터 없으면 자동 생성 (games/*/generate_sft.py) → train → eval
bash eunbi/run_blotto.sh

# MGC 데이터 쓰려면:
EUNBI_DATA_SOURCE=mgc bash eunbi/run_blotto.sh

# 빠른 스모크:
bash eunbi/run_smoke.sh
```

Swap `run_blotto.sh` for `run_mafia.sh` / `run_ipd.sh` / `run_codenames.sh`, or use `bash eunbi/run_all.sh` for all four.

Full pipelines (data prep + above): `bash mgc2025_sft/run_*.sh` or `bash rlgaming/run_*.sh` — see [README.md](../README.md).
