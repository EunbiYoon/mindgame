"""Game registry and eval scoring for RLGaming pipeline."""

from __future__ import annotations

from typing import Any

from eval.metrics import (
    blotto_allocation_match,
    blotto_allocation_valid,
    codenames_action_match,
    codenames_action_valid,
    codenames_json_validity,
    extract_json,
    ipd_choice_valid,
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


def _observation_text(ex: dict) -> str:
    prompt = ex.get("prompt", "")
    if "OBSERVATION:" in prompt:
        return prompt.split("OBSERVATION:", 1)[1]
    return prompt


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


def score_mafia_sft(pred_text: str, ex: dict) -> dict:
    obj = extract_json(pred_text)
    valid = json_validity(obj)
    gold_vote = (ex.get("output") or {}).get("action", {}).get("vote")
    action = obj.get("action", {}) if isinstance(obj, dict) else {}
    pred_vote = action.get("vote") if isinstance(action, dict) else None
    vote_ok = vote_accuracy(pred_vote, gold_vote)
    return {
        "format": "sft_json",
        "valid_json": valid,
        "vote_ok": vote_ok,
        "action_valid": valid,
        "action_match": valid and vote_ok,
        "exact": valid and vote_ok,
        "contains": valid and vote_ok,
        "nonempty": int(bool(pred_text)),
    }


def score_mgc_action(pred_text: str, ex: dict, game: str) -> dict:
    gold = ex.get("completion", "")
    obs = _observation_text(ex)
    obj = extract_json(pred_text)
    exact = action_exact_match(pred_text, gold)
    contains = action_contains_match(pred_text, gold)
    action_valid = 0
    action_match = 0

    if game == "blotto":
        action_valid = blotto_allocation_valid(obj) if obj else 0
        action_match = blotto_allocation_match(obj, gold) if obj else int(contains)
    elif game == "ipd":
        phase = (ex.get("meta") or {}).get("phase") or detect_ipd_phase(obs)
        if phase == "communication":
            action_valid = 1
            action_match = int(exact or contains)
        else:
            action_valid = ipd_choice_valid(obj) if obj else 0
            action_match = int(exact or contains)
    elif game == "codenames":
        role = (ex.get("meta") or {}).get("role") or detect_codenames_role(obs)
        board = _board_words(obs)
        action_valid = codenames_action_valid(obj, role, board) if obj else 0
        gold_action = {}
        if isinstance(ex.get("output"), dict):
            action = ex["output"].get("action", {})
            if isinstance(action, dict):
                gold_action = action
        action_match = (
            codenames_action_match(obj, gold_action, role)
            if obj and gold_action
            else int(contains)
        )
    else:
        action_valid = json_validity(obj) if obj else 0
        action_match = int(contains)

    return {
        "format": "mgc_action",
        "valid_json": json_validity(obj) if obj else 0,
        "vote_ok": 0,
        "action_valid": action_valid,
        "action_match": action_match,
        "exact": int(exact),
        "contains": int(contains),
        "nonempty": int(bool(pred_text)),
    }


def score_example(game: str, pred_text: str, ex: dict) -> dict:
    meta = ex.get("meta") or {}
    source = meta.get("source", "")
    if game == "mafia" and (source == "rule_based" or "output" in ex):
        return score_mafia_sft(pred_text, ex)
    return score_mgc_action(pred_text, ex, game)


def aggregate_metrics(game: str, totals: dict) -> dict:
    n = max(totals["n"], 1)
    metrics = {
        "nonempty_rate": round(totals["nonempty"] / n, 4),
        "json_valid_rate": round(totals["json_valid"] / n, 4),
        "action_exact_match": round(totals["exact"] / n, 4),
        "action_contains_match": round(totals["contains"] / n, 4),
        "action_valid_rate": round(totals["action_valid"] / n, 4),
        "action_match_rate": round(totals["action_match"] / n, 4),
    }
    if game == "mafia":
        metrics["vote_accuracy"] = round(totals["vote_ok"] / n, 4)
    return metrics
