"""Run output dirs: eunbi/lora/runs/<id>/, eunbi/eval/runs/<id>/."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "games"
DATA_RUNS_DIR = GAME_DIR / "runs"
LORA_DIR = PROJECT_ROOT / "eunbi" / "lora"
LORA_RUNS_DIR = LORA_DIR / "runs"
EVAL_DIR = PROJECT_ROOT / "eunbi" / "eval"
EVAL_RUNS_DIR = EVAL_DIR / "runs"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_run_id(run_id: str | None = None) -> str:
    return run_id or os.environ.get("RUN_ID") or utc_stamp()


def new_data_run_dir(run_id: str | None = None) -> Path:
    rid = resolve_run_id(run_id)
    run_dir = DATA_RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def new_lora_run_dir(run_id: str | None = None) -> Path:
    rid = resolve_run_id(run_id)
    run_dir = LORA_RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def new_eval_run_dir(run_id: str | None = None) -> Path:
    rid = resolve_run_id(run_id)
    run_dir = EVAL_RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_latest_pointer(base_dir: Path, run_dir: Path) -> None:
    rel = run_dir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    payload = {
        "run_id": run_dir.name,
        "path": rel,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (base_dir / "latest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def read_latest_path(base_dir: Path) -> Path | None:
    latest = base_dir / "latest.json"
    if not latest.is_file():
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    path = data.get("path")
    if not path:
        return None
    p = PROJECT_ROOT / path
    return p if p.is_dir() else None
