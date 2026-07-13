#!/usr/bin/env python3
"""Evaluate RLGaming LoRA on MindGames test jsonl."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eunbi.eval.model_utils import generate_text, load_model_and_tokenizer  # noqa: E402
from rlgaming.eval.games import GAMES, GAME_TITLES, aggregate_metrics, score_example  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="RLGaming MindGames eval")
    ap.add_argument("--game", default="mafia", choices=list(GAMES))
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    game = args.game
    run_dir = Path(args.out).parent if args.out else Path(args.test_file).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else run_dir / "metrics.json"

    model, tok, base_name = load_model_and_tokenizer(args.model_dir)

    totals = {
        "n": 0,
        "nonempty": 0,
        "json_valid": 0,
        "vote_ok": 0,
        "exact": 0,
        "contains": 0,
        "action_valid": 0,
        "action_match": 0,
        "sft_n": 0,
        "mgc_n": 0,
    }
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            ex = json.loads(line)
            pred_text = generate_text(model, tok, ex["prompt"], max_new_tokens=args.max_new_tokens).strip()
            row = score_example(game, pred_text, ex)
            totals["n"] += 1
            totals["nonempty"] += row["nonempty"]
            totals["json_valid"] += row["valid_json"]
            totals["vote_ok"] += row["vote_ok"]
            totals["exact"] += row["exact"]
            totals["contains"] += row["contains"]
            totals["action_valid"] += row["action_valid"]
            totals["action_match"] += row["action_match"]
            if row["format"] == "sft_json":
                totals["sft_n"] += 1
            else:
                totals["mgc_n"] += 1

    metrics = {
        "approach": "RLGaming",
        "game": GAME_TITLES[game],
        "game_key": game,
        "model_dir": args.model_dir,
        "base_model": base_name,
        "test_file": args.test_file,
        "n": totals["n"],
        "sft_examples": totals["sft_n"],
        "mgc_examples": totals["mgc_n"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        **totals,
        **aggregate_metrics(game, totals),
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
