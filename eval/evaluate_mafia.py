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

from eval.metrics import extract_json, json_validity, vote_accuracy
from eval.mindgames_tables import (
    table15_mafia_row,
    table5_error_row,
    write_result_md,
    write_tables,
)
from eval.model_utils import generate_text, load_model_and_tokenizer
from run_paths import EVAL_DIR, LORA_DIR, new_eval_run_dir, read_latest_path, write_latest_pointer


def main():
    ap = argparse.ArgumentParser(description="Secret Mafia eval (MindGames Table 15 / Table 5 format)")
    ap.add_argument("--model_dir", default=None, help="Default: lora/latest.json")
    ap.add_argument("--test_file", default="games/mafia/sft.jsonl")
    ap.add_argument("--n", type=int, default=100, help="Decision steps to evaluate")
    ap.add_argument("--model_name", default=None)
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

    total = valid = vote_ok = 0
    with open(args.test_file, encoding="utf-8") as f:
        for line in itertools.islice(f, args.n):
            ex = json.loads(line)
            pred_text = generate_text(model, tok, ex["prompt"])
            obj = extract_json(pred_text)
            valid += json_validity(obj)
            gold_vote = ex["output"]["action"]["vote"]
            pred_vote = obj.get("action", {}).get("vote") if isinstance(obj, dict) else None
            vote_ok += vote_accuracy(pred_vote, gold_vote)
            total += 1

    caused = total - valid
    clean = valid
    # Proxy: correct vote on valid JSON steps counts as a "win" for local Table 15-style summary.
    proxy_wins = vote_ok

    table15 = {
        "table": 15,
        "title": "Secret Mafia rankings proxy (MindGames Appendix; step-level vote accuracy)",
        "columns": ["Div", "R", "Model", "Team", "G", "W%", "Clean%", "Q"],
        "note": "W% is vote accuracy on valid JSON steps, not full 6-player match win rate.",
        "rows": [
            table15_mafia_row(
                model=model_label,
                games=total,
                wins=proxy_wins,
                clean_games=clean,
                min_games=50,
            )
        ],
    }
    table5 = {
        "table": 5,
        "title": "Error statistics (MindGames Table 5 format)",
        "columns": ["Environment", "Rank", "Model", "Games", "Clean", "Caused", "Witnessed", "Self-Forf.", "Opp-Forf."],
        "rows": [
            table5_error_row(
                environment="Secret Mafia",
                model=model_label,
                games=total,
                clean=clean,
                caused=caused,
            )
        ],
    }

    metrics = {
        "game": "Secret Mafia",
        "model_dir": str(model_dir),
        "base_model": base_name,
        "n_steps": total,
        "json_valid_rate": round(valid / max(total, 1), 4),
        "vote_accuracy": round(vote_ok / max(total, 1), 4),
        "clean_steps": clean,
        "clean_rate": round(100.0 * clean / max(total, 1), 1),
        "invalid_steps": caused,
        "error_rate": round(100.0 * caused / max(total, 1), 1),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_tables(run_dir, {"table15": table15, "table5": table5})
    write_result_md(
        run_dir,
        [
            "# Secret Mafia eval",
            "",
            f"- Model: `{model_label}` (`{model_dir}`)",
            f"- Base: `{base_name}`",
            f"- Steps: **{total}** | JSON valid: **{metrics['json_valid_rate']:.1%}** | Vote acc: **{metrics['vote_accuracy']:.1%}**",
            "",
            "See `tables/table15.json` and `tables/table5.json`.",
        ],
    )

    print(json.dumps(metrics, indent=2))
    if auto_run:
        write_latest_pointer(EVAL_DIR, run_dir)
        info = {
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "game": "Secret Mafia",
            "model_dir": str(model_dir),
            "base_model": base_name,
            "test_file": args.test_file,
            "n": args.n,
            "metrics": str(metrics_path),
            "tables": str(run_dir / "tables"),
            **metrics,
        }
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path} and tables/ under {run_dir}")


if __name__ == "__main__":
    main()
