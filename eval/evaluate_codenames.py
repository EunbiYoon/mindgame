#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (
    codenames_action_match,
    codenames_action_valid,
    codenames_json_validity,
    extract_json,
)
from eval.model_utils import generate_text, load_model_and_tokenizer
from run_paths import EVAL_DIR, LORA_DIR, new_eval_run_dir, read_latest_path, write_latest_pointer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="Default: lora/latest.json")
    ap.add_argument("--test_file", default="games/codenames/sft.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    if args.model_dir is None:
        latest = read_latest_path(LORA_DIR)
        if latest is None:
            raise SystemExit("No --model_dir and no lora/latest.json. Train LoRA first or pass --model_dir.")
        args.model_dir = str(latest)

    if args.out is None:
        run_dir = new_eval_run_dir(args.run_id)
        args.out = str(run_dir / "metrics_codenames.json")
        auto_run = True
    else:
        run_dir = Path(args.out).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_run = False

    model, tok, base_name = load_model_and_tokenizer(args.model_dir)

    total = valid_json = valid_action = action_match = 0
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            ex = json.loads(line)
            pred_text = generate_text(model, tok, ex["prompt"], max_new_tokens=256)
            obj = extract_json(pred_text)
            role = ex["input"]["game_state"]["role"]
            board_words = [c["word"] for c in ex["input"]["game_state"].get("board", [])]
            valid_json += codenames_json_validity(obj, role)
            valid_action += codenames_action_valid(obj, role, board_words)
            gold = ex["output"]["action"]
            action_match += codenames_action_match(obj, gold, role)
            total += 1

    metrics = {
        "game": "Codenames",
        "model_dir": args.model_dir,
        "base_model": base_name,
        "n": total,
        "json_valid_rate": valid_json / max(total, 1),
        "action_valid_rate": valid_action / max(total, 1),
        "action_exact_match": action_match / max(total, 1),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    if auto_run:
        write_latest_pointer(EVAL_DIR, run_dir)
        info = {
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "game": "Codenames",
            "model_dir": args.model_dir,
            "test_file": args.test_file,
            "n": args.n,
            "metrics": args.out,
            **metrics,
        }
        (run_dir / "run_info_codenames.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
