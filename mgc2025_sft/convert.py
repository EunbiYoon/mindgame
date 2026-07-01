#!/usr/bin/env python3
"""Build train/test SFT jsonl from MGC2025 HuggingFace trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_cache import ensure_hf_cache  # noqa: E402
from mgc2025_sft.lib import GAMES, is_test_row, trajectory_examples  # noqa: E402


def load_rows(hf_config: str, max_trajectories: int, cache_dir: str | None):
    from datasets import load_dataset

    kwargs: dict = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if max_trajectories > 0:
        ds = load_dataset(
            "mindgameschallenge/MGC2025",
            hf_config,
            split=f"train[:{max_trajectories}]",
            **kwargs,
        )
        yield from ds
        return

    ds = load_dataset(
        "mindgameschallenge/MGC2025",
        hf_config,
        split="train",
        streaming=True,
        **kwargs,
    )
    for row in ds:
        yield row


def convert_game(
    game: str,
    out_dir: Path,
    *,
    max_trajectories: int,
    test_frac: float,
    seed: int,
    cache_dir: str | None,
) -> dict[str, int]:
    cfg = GAMES[game]
    train_path = out_dir / f"{game}_train.jsonl"
    test_path = out_dir / f"{game}_test.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "test": 0, "trajectories": 0, "skipped_empty": 0}
    pending_train: list[dict] = []
    pending_test: list[dict] = []

    for row in load_rows(cfg["hf_config"], max_trajectories, cache_dir):
        counts["trajectories"] += 1
        row_id = row.get("player_game_id") or row.get("game_id")
        examples = list(trajectory_examples(game, row))
        if not examples:
            counts["skipped_empty"] += 1
            continue
        bucket = pending_test if is_test_row(row_id, test_frac=test_frac, seed=seed) else pending_train
        bucket.extend(examples)

    if not pending_train and pending_test:
        # Small-sample fallback: keep at least one trajectory for training.
        split_at = max(1, int(len(pending_test) * (1.0 - test_frac)))
        pending_train = pending_test[:split_at]
        pending_test = pending_test[split_at:]

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        for ex in pending_train:
            train_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts["train"] += 1
        for ex in pending_test:
            test_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts["test"] += 1

    info = {
        "game": game,
        "hf_config": cfg["hf_config"],
        "max_trajectories": max_trajectories,
        "test_frac": test_frac,
        "seed": seed,
        **counts,
        "train_file": str(train_path),
        "test_file": str(test_path),
    }
    (out_dir / f"{game}_convert.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MGC2025 trajectories to SFT jsonl")
    ap.add_argument(
        "--game",
        default="all",
        choices=["all", *GAMES.keys()],
        help="Which game config to convert",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for *_train.jsonl and *_test.jsonl",
    )
    ap.add_argument(
        "--max_trajectories",
        type=int,
        default=0,
        help="Cap trajectories per game (0 = full dataset via streaming)",
    )
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache dir")
    args = ap.parse_args()
    hf_datasets_cache = ensure_hf_cache()
    if args.cache_dir is None:
        args.cache_dir = hf_datasets_cache

    out_dir = Path(args.out_dir)
    games = list(GAMES.keys()) if args.game == "all" else [args.game]
    summary: dict[str, dict] = {}
    for game in games:
        summary[game] = convert_game(
            game,
            out_dir,
            max_trajectories=args.max_trajectories,
            test_frac=args.test_frac,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    (out_dir / "convert_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
