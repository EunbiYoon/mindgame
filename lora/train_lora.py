#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_cache import ensure_hf_cache
from run_paths import LORA_DIR, new_lora_run_dir, write_latest_pointer

def format_example(ex):
    return ex["prompt"] + "\n\nANSWER:\n" + ex["completion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="games/mafia/sft.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out_dir", default=None, help="Default: lora/runs/<timestamp>/")
    ap.add_argument("--run_id", default=None, help="Run folder name (default: RUN_ID env or UTC timestamp)")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()
    hf_datasets_cache = ensure_hf_cache()

    if args.out_dir is None:
        run_dir = new_lora_run_dir(args.run_id)
        args.out_dir = str(run_dir)
    else:
        run_dir = Path(args.out_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise SystemExit(
            f"Train file not found: {train_path.resolve()}\n"
            "Generate data first, e.g.:\n"
            "  python games/mafia/generate_sft.py --out games/mafia/sft.jsonl\n"
            "  python games/blotto/generate_sft.py --out games/blotto/sft.jsonl\n"
            "  python games/ipd/generate_sft.py --out games/ipd/sft.jsonl\n"
            "  python games/codenames/generate_sft.py --out games/codenames/sft.jsonl\n"
            "Or run a pipeline, e.g.:\n"
            "  bash mgc2025_sft/run_mafia.sh\n"
            "  bash rlgaming/run_mafia.sh"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    )
    if args.load_in_4bit:
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
    model.print_trainable_parameters()

    ds = load_dataset(
        "json", data_files=args.train_file, split="train", cache_dir=hf_datasets_cache
    )
    ds = ds.map(lambda x: {"text": format_example(x)})

    training_args = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    write_latest_pointer(LORA_DIR, run_dir)
    info = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": args.out_dir,
        "train_file": args.train_file,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "load_in_4bit": args.load_in_4bit,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter to {args.out_dir}")

if __name__ == "__main__":
    main()
