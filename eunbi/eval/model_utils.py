"""Load LoRA checkpoints saved under eunbi/lora/runs/<id>/ or HuggingFace hub ids."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _hub_cache_dir() -> str | None:
    return os.environ.get("HUGGINGFACE_HUB_CACHE")


def _is_hub_id(ref: str) -> bool:
    """True for HuggingFace repo ids like Qwen/Qwen3-8B (exactly one slash)."""
    if ref.startswith(("/", "./", "../")):
        return False
    parts = ref.split("/")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


def _resolve_model_ref(model_dir: str | Path) -> tuple[Path | str, bool]:
    """Return (local path or hub id, is_local_dir)."""
    path = Path(model_dir)
    if path.is_dir():
        return path, True
    ref = str(model_dir)
    if _is_hub_id(ref):
        return ref, False
    raise FileNotFoundError(f"Model dir not found: {model_dir}")


def load_model_and_tokenizer(model_dir: str | Path, *, load_in_4bit: bool | None = None):
    model_ref, is_local = _resolve_model_ref(model_dir)
    cache_dir = _hub_cache_dir()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if load_in_4bit is None:
        load_in_4bit = os.environ.get("MAF_LORA_4BIT", "0") == "1"

    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if is_local:
        local_dir = Path(model_ref)
        adapter_cfg_path = local_dir / "adapter_config.json"
        tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if adapter_cfg_path.is_file():
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-8B")
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype,
                quantization_config=quant,
                cache_dir=cache_dir,
            )
            model = PeftModel.from_pretrained(base, str(local_dir))
        else:
            base_name = str(local_dir)
            model = AutoModelForCausalLM.from_pretrained(
                str(local_dir),
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype,
                quantization_config=quant,
            )
    else:
        base_name = str(model_ref)
        tokenizer = AutoTokenizer.from_pretrained(
            base_name, trust_remote_code=True, cache_dir=cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quant,
            cache_dir=cache_dir,
        )

    model.eval()
    return model, tokenizer, base_name


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 384) -> str:
    text = prompt + "\n\nANSWER:\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
