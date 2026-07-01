#!/usr/bin/env python3
"""Evaluate STARS agent on MindGames test trajectories (Ollama / Qwen3-8B)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stars.agent.agent import StarsMafiaAgent  # noqa: E402
from stars.agent.game_agent import StarsGameAgent, dry_run_action  # noqa: E402
from stars.games import (  # noqa: E402
    GAMES,
    GAME_TITLES,
    aggregate_metrics,
    game_key,
    score_example,
)
from stars.ollama_client import OllamaClient  # noqa: E402


def _mafia_dry_run(ex: dict) -> tuple[str, bool, list]:
    from stars.agent.belief_state import BeliefState
    from stars.parse.observation import parse_prompt

    parsed = parse_prompt(ex["prompt"])
    belief = BeliefState()
    belief.update_from_parsed(parsed)
    suspect = belief.top_suspect() or (belief.valid_targets[:1] or ["0"])[0]
    pred_text = json.dumps(
        {
            "reasoning": "dry-run: vote highest P(mafia)",
            "action": {"speak": "", "vote": suspect, "raw": f"[{suspect}]"},
        }
    )
    return pred_text, True, []


def main() -> None:
    ap = argparse.ArgumentParser(description="STARS MindGames eval (no fine-tuning)")
    ap.add_argument("--game", required=True, choices=list(GAMES))
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--ollama_host", default=None)
    ap.add_argument("--ollama_model", default=None)
    ap.add_argument("--max_react", type=int, default=3)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--dry_run", action="store_true", help="Skip Ollama; heuristic fallback for smoke test")
    args = ap.parse_args()

    game = args.game
    game_title = GAME_TITLES[game]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectories.jsonl"
    metrics_path = run_dir / "metrics.json"

    client = OllamaClient(host=args.ollama_host, model=args.ollama_model)
    if not args.dry_run and not client.ping():
        raise SystemExit(
            f"Ollama not reachable at {client.host}. "
            f"Start Ollama and pull model: ollama pull {client.model}"
        )

    if game == "mafia":
        agent = StarsMafiaAgent(
            client=client,
            max_react_steps=args.max_react,
            max_retries=args.max_retries,
        )
    else:
        agent = StarsGameAgent(
            game=game,
            client=client,
            max_react_steps=args.max_react,
            max_retries=args.max_retries,
        )

    examples: list[dict] = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            examples.append(json.loads(line))

    by_game: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_game[game_key(ex)].append(i)

    totals = {
        "n": 0,
        "valid_actions": 0,
        "json_valid": 0,
        "vote_match": 0,
        "exact_match": 0,
        "contains_match": 0,
        "action_valid": 0,
        "action_match": 0,
    }

    with traj_path.open("w", encoding="utf-8") as traj_out:
        for gkey, indices in by_game.items():
            for idx in indices:
                ex = examples[idx]
                if args.dry_run:
                    if game == "mafia":
                        pred_text, result_valid, react_steps = _mafia_dry_run(ex)
                    else:
                        pred_text = dry_run_action(game, ex["prompt"], ex.get("meta"))
                        result_valid = True
                        react_steps = []
                elif game == "mafia":
                    out = agent.act(ex["prompt"], game_key=gkey)
                    pred_text = out.action_text
                    result_valid = out.valid
                    react_steps = out.react_steps
                else:
                    out = agent.act(ex["prompt"], game_key=gkey, meta=ex.get("meta"))
                    pred_text = out.action_text
                    result_valid = out.valid
                    react_steps = out.react_steps

                scores = score_example(game, pred_text, ex)
                totals["n"] += 1
                totals["valid_actions"] += int(result_valid)
                totals["json_valid"] += scores["json_valid"]
                totals["vote_match"] += scores["vote_match"]
                totals["exact_match"] += scores["exact_match"]
                totals["contains_match"] += scores["contains_match"]
                totals["action_valid"] += scores["action_valid"]
                totals["action_match"] += scores["action_match"]

                record = {
                    "index": idx,
                    "game_key": gkey,
                    "meta": ex.get("meta"),
                    "valid_action": result_valid,
                    "prediction": pred_text,
                    "gold_completion": ex.get("completion"),
                    "gold_output": ex.get("output"),
                    "scores": scores,
                    "react_steps": react_steps,
                }
                traj_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not args.dry_run and len(indices) > 1:
                summary = "\n".join(
                    f"turn {examples[i].get('meta', {}).get('turn', i)}: {examples[i].get('completion', '')[:120]}"
                    for i in indices[:8]
                )
                if game == "mafia":
                    agent.post_game_analysis(gkey, summary)
                else:
                    agent.post_game_analysis(gkey, summary, game_title=game_title)

    metrics = {
        "approach": "STARS",
        "game": game_title,
        "game_key": game,
        "model": client.model,
        "inference": "ollama",
        "fine_tuning": False,
        "test_file": args.test_file,
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **totals,
        **aggregate_metrics(game, totals),
        "lessons": agent._game_lessons,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    latest = ROOT / "stars" / "latest.json"
    rel_path = run_dir.relative_to(ROOT).as_posix() if str(run_dir).startswith(str(ROOT)) else str(run_dir)
    latest.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "game": game,
                "path": rel_path,
                "updated_at": metrics["created_at"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path} and {traj_path}")


if __name__ == "__main__":
    main()
