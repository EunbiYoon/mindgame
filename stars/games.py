"""Game registry and scoring for STARS evaluation."""

from __future__ import annotations

from typing import Any, Callable

from eunbi.eval.metrics import (
    blotto_allocation_match,
    blotto_allocation_valid,
    codenames_action_match,
    codenames_action_valid,
    codenames_json_validity,
    extract_json,
    ipd_choice_match,
    ipd_choice_valid,
    ipd_json_validity,
    json_validity,
    vote_accuracy,
)
from mgc2025_sft.lib import action_contains_match, action_exact_match, detect_codenames_role, detect_ipd_phase

GAMES = ("mafia", "blotto", "ipd", "codenames")

GAME_TITLES = {
    "mafia": "Secret Mafia",
    "blotto": "Colonel Blotto",
    "ipd": "Three-Player IPD",
    "codenames": "Codenames",
}


def game_meta(ex: dict) -> dict[str, Any]:
    return ex.get("meta") or {}


def game_key(ex: dict) -> str:
    meta = game_meta(ex)
    gid = meta.get("game_id")
    if gid is not None:
        return f"mgc:{gid}"
    return f"sft:{hash(ex.get('prompt', ''))}"


def _gold_vote(ex: dict) -> str | None:
    if "output" in ex:
        return str(ex["output"].get("action", {}).get("vote", ""))
    return None


def _ipd_phase(ex: dict) -> str:
    meta = game_meta(ex)
    if meta.get("phase"):
        return str(meta["phase"])
    obs = ex.get("prompt", "").split("OBSERVATION:", 1)[-1]
    return detect_ipd_phase(obs)


def _codenames_role(ex: dict) -> str:
    meta = game_meta(ex)
    if meta.get("role"):
        return str(meta["role"])
    obs = ex.get("prompt", "").split("OBSERVATION:", 1)[-1]
    return detect_codenames_role(obs)


def _board_words(observation: str) -> list[str]:
    words: list[str] = []
    in_board = False
    for line in observation.splitlines():
        if "codenames words" in line.lower():
            in_board = True
            continue
        if not in_board:
            continue
        token = line.strip().split()
        if token:
            words.append(token[0].lower())
    return words


def score_mafia(pred_text: str, ex: dict) -> dict:
    gold_vote = _gold_vote(ex)
    if gold_vote is not None:
        obj = extract_json(pred_text)
        action = obj.get("action", {}) if isinstance(obj, dict) else {}
        pred_vote = action.get("vote") if isinstance(action, dict) else None
        return {
            "json_valid": json_validity(obj),
            "vote_match": vote_accuracy(pred_vote, gold_vote),
            "exact_match": 0,
            "contains_match": 0,
            "action_valid": 0,
            "action_match": 0,
        }
    gold = ex.get("completion", "")
    return {
        "json_valid": 0,
        "vote_match": 0,
        "exact_match": int(action_exact_match(pred_text, gold)),
        "contains_match": int(action_contains_match(pred_text, gold)),
        "action_valid": 0,
        "action_match": 0,
    }


def score_blotto(pred_text: str, ex: dict) -> dict:
    gold = ex.get("completion", "")
    obj = extract_json(pred_text)
    valid = blotto_allocation_valid(obj) if obj else 0
    match = blotto_allocation_match(obj, gold) if obj else 0
    return {
        "json_valid": json_validity(obj) if obj else 0,
        "vote_match": 0,
        "exact_match": int(action_exact_match(pred_text, gold)),
        "contains_match": int(action_contains_match(pred_text, gold)),
        "action_valid": valid,
        "action_match": match,
    }


def score_ipd(pred_text: str, ex: dict) -> dict:
    gold = ex.get("completion", "")
    phase = _ipd_phase(ex)
    if phase == "communication":
        return {
            "json_valid": 0,
            "vote_match": 0,
            "exact_match": int(action_exact_match(pred_text, gold)),
            "contains_match": int(action_contains_match(pred_text, gold)),
            "action_valid": 1,
            "action_match": int(action_exact_match(pred_text, gold)),
        }
    obj = extract_json(pred_text)
    gold_choice = ""
    if isinstance(ex.get("output"), dict):
        action = ex["output"].get("action", {})
        if isinstance(action, dict):
            gold_choice = str(action.get("choice", ""))
    return {
        "json_valid": ipd_json_validity(obj),
        "vote_match": 0,
        "exact_match": int(action_exact_match(pred_text, gold)),
        "contains_match": int(action_contains_match(pred_text, gold)),
        "action_valid": ipd_choice_valid(obj),
        "action_match": ipd_choice_match(obj, gold_choice) if gold_choice else int(action_exact_match(pred_text, gold)),
    }


def score_codenames(pred_text: str, ex: dict) -> dict:
    gold = ex.get("completion", "")
    role = _codenames_role(ex)
    obs = ex.get("prompt", "").split("OBSERVATION:", 1)[-1]
    board_words = _board_words(obs)
    obj = extract_json(pred_text)
    gold_action: dict[str, Any] = {}
    if isinstance(ex.get("output"), dict):
        action = ex["output"].get("action", {})
        if isinstance(action, dict):
            gold_action = action
    return {
        "json_valid": codenames_json_validity(obj, role) if obj else 0,
        "vote_match": 0,
        "exact_match": int(action_exact_match(pred_text, gold)),
        "contains_match": int(action_contains_match(pred_text, gold)),
        "action_valid": codenames_action_valid(obj, role, board_words) if obj else 0,
        "action_match": codenames_action_match(obj, gold_action, role) if obj and gold_action else int(action_contains_match(pred_text, gold)),
    }


SCORERS: dict[str, Callable[[str, dict], dict]] = {
    "mafia": score_mafia,
    "blotto": score_blotto,
    "ipd": score_ipd,
    "codenames": score_codenames,
}


def score_example(game: str, pred_text: str, ex: dict) -> dict:
    return SCORERS[game](pred_text, ex)


def aggregate_metrics(game: str, totals: dict) -> dict:
    n = max(totals["n"], 1)
    out = {
        "valid_action_rate": round(totals["valid_actions"] / n, 4),
        "json_valid_rate": round(totals["json_valid"] / n, 4),
        "vote_accuracy": round(totals["vote_match"] / n, 4),
        "action_exact_match": round(totals["exact_match"] / n, 4),
        "action_contains_match": round(totals["contains_match"] / n, 4),
        "action_valid_rate": round(totals["action_valid"] / n, 4),
        "action_match_rate": round(totals["action_match"] / n, 4),
    }
    if game == "mafia":
        out.pop("action_valid_rate", None)
        out.pop("action_match_rate", None)
    if game != "mafia":
        out.pop("vote_accuracy", None)
    return out
