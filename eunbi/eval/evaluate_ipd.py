#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eunbi.eval.metrics import (
    extract_json,
    ipd_choice_match,
    ipd_choice_valid,
    ipd_json_validity,
    ipd_simulated_reward,
)
from eunbi.eval.model_utils import generate_text, load_model_and_tokenizer
from run_paths import EVAL_DIR, LORA_DIR, new_eval_run_dir, read_latest_path, write_latest_pointer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="Default: eunbi/lora/latest.json")
    ap.add_argument("--test_file", default="games/ipd/sft.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    if args.model_dir is None:
        latest = read_latest_path(LORA_DIR)
        if latest is None:
            raise SystemExit("No --model_dir and no eunbi/lora/latest.json. Train LoRA first or pass --model_dir.")
        args.model_dir = str(latest)

    if args.out is None:
        run_dir = new_eval_run_dir(args.run_id)
        args.out = str(run_dir / "metrics_ipd.json")
        auto_run = True
    else:
        run_dir = Path(args.out).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_run = False

    model, tok, base_name = load_model_and_tokenizer(args.model_dir)

    total = valid_json = valid_choice = choice_match = reward_ok = 0
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            ex = json.loads(line)
            if ex["input"]["game_state"].get("phase") != "decision":
                continue
            pred_text = generate_text(model, tok, ex["prompt"], max_new_tokens=256)
            obj = extract_json(pred_text)
            valid_json += ipd_json_validity(obj)
            valid_choice += ipd_choice_valid(obj)
            gold = ex["output"]["action"].get("choice", "")
            choice_match += ipd_choice_match(obj, gold)
            joint = ex.get("meta", {}).get("joint_actions", {})
            self_id = ex["input"]["game_state"]["self_id"]
            reward_ok += ipd_simulated_reward(obj, joint, self_id)
            total += 1

    metrics = {
        "game": "Three-Player IPD",
        "model_dir": args.model_dir,
        "base_model": base_name,
        "n": total,
        "json_valid_rate": valid_json / max(total, 1),
        "choice_valid_rate": valid_choice / max(total, 1),
        "choice_exact_match": choice_match / max(total, 1),
        "simulated_round_score_match": reward_ok / max(total, 1),
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
            "game": "Three-Player IPD",
            "model_dir": args.model_dir,
            "test_file": args.test_file,
            "n": args.n,
            "metrics": args.out,
            **metrics,
        }
        (run_dir / "run_info_ipd.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
