"""MindGames paper table layouts (local run: your model only, no paper reference rows)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / max(d, 1), 1)


def table12_blotto_row(
    *,
    model: str,
    team: str = "local",
    games: int,
    wins: int,
    clean_games: int,
    division: str = "Local",
    rank: int = 1,
    min_games: int = 30,
) -> dict[str, Any]:
    """MindGames Appendix Table 12 — Colonel Blotto rankings."""
    return {
        "Div": division,
        "R": rank,
        "Model": model,
        "Team": team,
        "G": games,
        "W%": _pct(wins, games),
        "Clean%": _pct(clean_games, games),
        "Q": games >= min_games,
    }


def table5_error_row(
    *,
    environment: str,
    model: str,
    games: int,
    clean: int,
    caused: int,
    witnessed: int = 0,
    self_forfeit: int | None = None,
    opp_forfeit: int | None = None,
    rank: int = 1,
) -> dict[str, Any]:
    """MindGames Table 5 — per-model error statistics."""
    if self_forfeit is None:
        self_forfeit = caused
    if opp_forfeit is None:
        opp_forfeit = witnessed
    return {
        "Environment": environment,
        "Rank": rank,
        "Model": model,
        "Games": games,
        "Clean": clean,
        "Caused": caused,
        "Witnessed": witnessed,
        "Self-Forf.": self_forfeit,
        "Opp-Forf.": opp_forfeit,
    }


def table15_mafia_row(
    *,
    model: str,
    team: str = "local",
    games: int,
    wins: int,
    clean_games: int,
    division: str = "Local",
    rank: int = 1,
    min_games: int = 50,
) -> dict[str, Any]:
    """MindGames Appendix Table 15 — Secret Mafia rankings (local proxy metrics)."""
    return {
        "Div": division,
        "R": rank,
        "Model": model,
        "Team": team,
        "G": games,
        "W%": _pct(wins, games),
        "Clean%": _pct(clean_games, games),
        "Q": games >= min_games,
    }


def table3_local_row(
    *,
    game: str,
    model: str,
    base_model: str,
    score: float,
    score_label: str,
    clean_pct: float,
    fine_tuning: bool = True,
    rank: int = 1,
) -> dict[str, Any]:
    """Local leaderboard row inspired by MindGames Table 3 (model × game summary)."""
    return {
        "R": rank,
        "Game": game,
        "Model": model,
        "Base": base_model,
        "Fine-tuning": fine_tuning,
        score_label: score,
        "Clean%": clean_pct,
    }


def table4_row(
    *,
    environment: str,
    model: str,
    games: int,
    clean_pct: float,
    error_pct: float,
    top_reason: str = "",
    second_reason: str = "",
) -> dict[str, Any]:
    """MindGames Table 4 — per-model error aggregation by environment."""
    return {
        "Environment": environment,
        "Model": model,
        "# Games": games,
        "Clean (%)": clean_pct,
        "Error (%)": error_pct,
        "Top Reason": top_reason,
        "2nd Reason": second_reason,
    }


def write_tables(run_dir: Path, tables: dict[str, Any]) -> None:
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in tables.items():
        path = table_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_result_md(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
