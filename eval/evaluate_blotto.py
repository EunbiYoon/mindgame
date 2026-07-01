#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.blotto_match import run_blotto_matches
from eval.mindgames_tables import (
    table12_blotto_row,
    table5_error_row,
    write_result_md,
    write_tables,
)
from eval.model_utils import generate_text, load_model_and_tokenizer
from run_paths import EVAL_DIR, LORA_DIR, new_eval_run_dir, read_latest_path, write_latest_pointer


def main():
    ap = argparse.ArgumentParser(description="Colonel Blotto eval (MindGames Table 12 / Table 5 format)")
    ap.add_argument("--model_dir", default=None, help="Default: lora/latest.json")
    ap.add_argument("--n_games", type=int, default=30, help="Full matches to play (paper qualification: >=30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default=None, help="Label in results table")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    if args.model_dir is None:
        latest = read_latest_path(LORA_DIR)
        if latest is None:
            raise SystemExit("No --model_dir and no lora/latest.json. Train LoRA first.")
        args.model_dir = str(latest)

    model_dir = Path(args.model_dir)
    model_label = args.model_name or model_dir.name

    if args.out is None:
        run_dir = new_eval_run_dir(args.run_id)
        auto_run = True
    else:
        run_dir = Path(args.out).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_run = False

    model, tok, base_name = load_model_and_tokenizer(args.model_dir)
    stats = run_blotto_matches(
        lambda prompt: generate_text(model, tok, prompt),
        n_games=args.n_games,
        seed=args.seed,
    )

    table12 = {
        "table": 12,
        "title": "Colonel Blotto rankings (MindGames Appendix E)",
        "columns": ["Div", "R", "Model", "Team", "G", "W%", "Clean%", "Q"],
        "rows": [
            table12_blotto_row(
                model=model_label,
                games=stats.games,
                wins=stats.wins,
                clean_games=stats.clean_games,
            )
        ],
    }
    table5 = {
        "table": 5,
        "title": "Error statistics (MindGames Table 5 format)",
        "columns": ["Environment", "Rank", "Model", "Games", "Clean", "Caused", "Witnessed", "Self-Forf.", "Opp-Forf."],
        "rows": [
            table5_error_row(
                environment="Colonel Blotto",
                model=model_label,
                games=stats.games,
                clean=stats.clean_games,
                caused=stats.caused_errors,
                self_forfeit=stats.caused_errors,
            )
        ],
    }

    metrics = {
        "game": "Colonel Blotto",
        "model_dir": str(model_dir),
        "base_model": base_name,
        "n_games": stats.games,
        "wins": stats.wins,
        "losses": stats.losses,
        "ties": stats.ties,
        "win_rate": round(100.0 * stats.wins / max(stats.games, 1), 1),
        "clean_games": stats.clean_games,
        "clean_rate": round(100.0 * stats.clean_games / max(stats.games, 1), 1),
        "caused_errors": stats.caused_errors,
        "error_rate": round(100.0 * stats.caused_errors / max(stats.games, 1), 1),
        "invalid_format": stats.invalid_format,
        "invalid_units": stats.invalid_units,
        "avg_rounds_per_match": round(stats.rounds_played / max(stats.games, 1), 2),
        "avg_round_wins_per_match": round(stats.round_wins / max(stats.games, 1), 2),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_tables(run_dir, {"table12": table12, "table5": table5})
    write_result_md(
        run_dir,
        [
            "# Colonel Blotto eval",
            "",
            f"- Model: `{model_label}` (`{model_dir}`)",
            f"- Base: `{base_name}`",
            f"- Games: **{stats.games}** | Win%: **{metrics['win_rate']}%** | Clean%: **{metrics['clean_rate']}%**",
            f"- Errors: **{stats.caused_errors}** (format {stats.invalid_format}, units {stats.invalid_units})",
            "",
            "See `tables/table12.json` (rankings) and `tables/table5.json` (errors).",
        ],
    )

    print(json.dumps(metrics, indent=2))
    if auto_run:
        write_latest_pointer(EVAL_DIR, run_dir)
        info = {
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "game": "Colonel Blotto",
            "model_dir": str(model_dir),
            "base_model": base_name,
            "n_games": args.n_games,
            "metrics": str(metrics_path),
            "tables": str(run_dir / "tables"),
            **metrics,
        }
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path} and tables/ under {run_dir}")


if __name__ == "__main__":
    main()
