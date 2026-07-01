#!/usr/bin/env python3
"""Build multi-source RLGaming SFT data for Secret Mafia."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_cache import ensure_hf_cache  # noqa: E402
from mgc2025_sft.convert import load_rows  # noqa: E402
from mgc2025_sft.lib import GAMES, is_test_row, trajectory_examples  # noqa: E402
from rlgaming.data.filter import dedupe_examples, filter_examples  # noqa: E402
from rlgaming.format.observation import build_rlgaming_prompt  # noqa: E402
from rlgaming.data.rule_agent import generate_rule_examples  # noqa: E402


def load_mgc_examples(
    game: str,
    *,
    max_trajectories: int,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in load_rows(GAMES[game]["hf_config"], max_trajectories, cache_dir):
        for ex in trajectory_examples(game, row):
            examples.append(ex)
    return examples


def apply_context_and_prompt(ex: dict[str, Any], game: str) -> dict[str, Any]:
    meta = ex.get("meta") or {}
    prompt = build_rlgaming_prompt(ex["prompt"], game, meta)
    out = dict(ex)
    out["prompt"] = prompt
    out.setdefault("meta", {})["context_optimized"] = True
    return out


def weighted_sample(
    examples: list[dict[str, Any]],
    *,
    rule_weight: int,
    mgc_weight: int,
    seed: int,
) -> list[dict[str, Any]]:
    if rule_weight <= 0 and mgc_weight <= 0:
        return examples
    buckets: dict[str, list[dict[str, Any]]] = {"rule_based": [], "mgc2025": []}
    for ex in examples:
        src = (ex.get("meta") or {}).get("source", "mgc2025")
        if src == "rule_based":
            buckets["rule_based"].append(ex)
        else:
            buckets["mgc2025"].append(ex)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for _ in range(rule_weight):
        out.extend(buckets["rule_based"])
    for _ in range(mgc_weight):
        out.extend(buckets["mgc2025"])
    if not out:
        return examples
    rng.shuffle(out)
    return out


def split_train_test(
    examples: list[dict[str, Any]],
    *,
    test_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for ex in examples:
        meta = ex.get("meta") or {}
        row_id = meta.get("player_game_id") or meta.get("game_id") or meta.get("self_role")
        if meta.get("source") == "rule_based":
            bucket = test if rng_bucket(row_id, test_frac, seed) else train
        else:
            bucket = test if is_test_row(row_id, test_frac=test_frac, seed=seed) else train
        bucket.append(ex)
    if not train and test:
        split_at = max(1, int(len(test) * (1.0 - test_frac)))
        train, test = test[:split_at], test[split_at:]
    return train, test


def rng_bucket(row_id, test_frac: float, seed: int) -> bool:
    return is_test_row(row_id, test_frac=test_frac, seed=seed)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="RLGaming multi-source SFT builder")
    ap.add_argument("--game", default="mafia", choices=["mafia", "blotto", "ipd", "codenames"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_rule_games", type=int, default=100)
    ap.add_argument("--mgc_max_trajectories", type=int, default=500)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--filter_win_only", action="store_true")
    ap.add_argument("--rule_weight", type=int, default=1)
    ap.add_argument("--mgc_weight", type=int, default=2)
    ap.add_argument("--skip_mgc", action="store_true")
    args = ap.parse_args()

    ensure_hf_cache()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    game = args.game

    raw_rule: list[dict[str, Any]] = []
    if game == "mafia" and args.n_rule_games > 0:
        raw_rule = generate_rule_examples(args.n_rule_games, args.seed)
    raw_mgc: list[dict] = []
    if not args.skip_mgc and args.mgc_max_trajectories > 0:
        raw_mgc = load_mgc_examples(
            game,
            max_trajectories=args.mgc_max_trajectories,
            cache_dir=args.cache_dir,
        )

    raw_all = raw_rule + raw_mgc
    filtered, filter_stats = filter_examples(
        iter(raw_all),
        win_only=args.filter_win_only,
    )
    filtered = dedupe_examples(filtered)
    mixed = weighted_sample(
        filtered,
        rule_weight=args.rule_weight,
        mgc_weight=args.mgc_weight,
        seed=args.seed,
    )
    optimized = [apply_context_and_prompt(ex, game) for ex in mixed]
    train, test = split_train_test(optimized, test_frac=args.test_frac, seed=args.seed)

    train_path = out_dir / f"{game}_train.jsonl"
    test_path = out_dir / f"{game}_test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    # Keep intermediate artifacts for debugging
    write_jsonl(out_dir / "raw_rule.jsonl", raw_rule)
    if raw_mgc:
        write_jsonl(out_dir / "raw_mgc.jsonl", raw_mgc[:5000])
    write_jsonl(out_dir / "filtered.jsonl", filtered)

    summary = {
        "approach": "RLGaming",
        "game": game,
        "n_rule_games": args.n_rule_games if game == "mafia" else 0,
        "mgc_max_trajectories": args.mgc_max_trajectories,
        "raw_rule_examples": len(raw_rule),
        "raw_mgc_examples": len(raw_mgc),
        "filter_stats": filter_stats,
        "after_dedupe": len(filtered),
        "after_mix": len(mixed),
        "train": len(train),
        "test": len(test),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "filter_win_only": args.filter_win_only,
        "rule_weight": args.rule_weight,
        "mgc_weight": args.mgc_weight,
    }
    (out_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
