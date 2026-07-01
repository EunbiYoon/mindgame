#!/usr/bin/env python3
"""Evaluate LoRA on MGC2025-derived test jsonl (raw action match)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.model_utils import generate_text, load_model_and_tokenizer  # noqa: E402
from mgc2025_sft.lib import GAMES, action_contains_match, action_exact_match  # noqa: E402
from run_paths import EVAL_DIR, LORA_DIR, new_eval_run_dir, read_latest_path, write_latest_pointer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="MGC2025 trajectory action-match eval")
    ap.add_argument("--game", required=True, choices=list(GAMES.keys()))
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.model_dir is None:
        latest = read_latest_path(LORA_DIR)
        if latest is None:
            raise SystemExit("No --model_dir and no lora/latest.json")
        args.model_dir = str(latest)

    if args.out is None:
        run_dir = new_eval_run_dir(args.run_id)
        args.out = str(run_dir / f"metrics_mgc_{args.game}.json")
        auto_run = True
    else:
        run_dir = Path(args.out).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_run = False

    model, tok, base_name = load_model_and_tokenizer(args.model_dir)

    total = exact = contains = nonempty = 0
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            ex = json.loads(line)
            pred = generate_text(model, tok, ex["prompt"], max_new_tokens=args.max_new_tokens)
            gold = ex["completion"]
            pred = pred.strip()
            if pred:
                nonempty += 1
            exact += int(action_exact_match(pred, gold))
            contains += int(action_contains_match(pred, gold))
            total += 1

    metrics = {
        "game": GAMES[args.game]["title"],
        "data_source": "mgc2025",
        "model_dir": args.model_dir,
        "base_model": base_name,
        "test_file": args.test_file,
        "n": total,
        "nonempty_rate": nonempty / max(total, 1),
        "action_exact_match": exact / max(total, 1),
        "action_contains_match": contains / max(total, 1),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    Path(args.out).write_text(text + "\n", encoding="utf-8")

    if auto_run:
        write_latest_pointer(EVAL_DIR, run_dir)
        info = {
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
            "metrics_path": args.out,
        }
        (run_dir / f"run_info_mgc_{args.game}.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
