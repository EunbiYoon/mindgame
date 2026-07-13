# eunbi — shared LoRA train & eval

Combines the former root-level `lora/` and `eval/` into one module.  
Used by `mgc2025_sft` and `rlgaming`.

```text
eunbi/
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

## Train & eval

```bash
bash mgc2025_sft/run_blotto.sh
bash rlgaming/run_blotto.sh
```

See [README.md](../README.md) for per-module commands.
