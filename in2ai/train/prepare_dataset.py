#!/usr/bin/env python3
"""Prepare GRPO dataset from In2AI rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_rollout_steps(path: Path) -> list[dict]:
    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("eligible") and row.get("prompt"):
                steps.append(row)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    steps = load_rollout_steps(Path(args.rollout_file))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in steps:
            record = {
                "prompt": row["prompt"],
                "credit": row.get("credit", row.get("normalized_reward", 0.0)),
                "game": row["game"],
                "env_state": row.get("env_state", {}),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(steps)} eligible prompts to {out_path}")


if __name__ == "__main__":
    main()
