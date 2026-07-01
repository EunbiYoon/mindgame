"""Output validation for structured MindGames actions."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    blotto_allocation_valid,
    codenames_action_valid,
    extract_json,
    ipd_choice_valid,
    mafia_action_valid,
    parse_allocation,
    parse_choice,
    parse_clue,
    parse_guess,
    parse_mafia_action,
)


@dataclass
class ValidationResult:
    valid: bool
    parsed: dict | str | None = None
    reason: str = ""


def validate_completion(game: str, completion: str, context: dict | None = None) -> ValidationResult:
    if not isinstance(context, dict):
        context = {}

    if game == "ipd" and context.get("phase", "decision") == "communication":
        return ValidationResult(True, completion, "communication_freeform")

    obj = extract_json(completion)
    if obj is None:
        return ValidationResult(False, reason="no_json")
    if not isinstance(obj, dict):
        return ValidationResult(False, parsed=obj, reason="invalid_format")

    if game == "blotto":
        if not blotto_allocation_valid(obj):
            alloc = None
            if isinstance(obj, dict):
                action = obj.get("action", {})
                if isinstance(action, dict):
                    alloc = parse_allocation(str(action.get("allocation", "")))
            reason = "invalid_units" if alloc else "invalid_format"
            return ValidationResult(False, obj, reason)
        alloc = parse_allocation(str(obj["action"]["allocation"]))
        return ValidationResult(True, alloc)

    if game == "ipd":
        if not ipd_choice_valid(obj):
            choice = None
            if isinstance(obj, dict):
                action = obj.get("action", {})
                if isinstance(action, dict):
                    choice = parse_choice(str(action.get("choice", "")))
            return ValidationResult(False, obj, "invalid_choice" if choice else "invalid_format")
        return ValidationResult(True, parse_choice(str(obj["action"]["choice"])))

    if game == "codenames":
        role = context.get("role", "operative")
        board_words = context.get("board_words", [])
        if not codenames_action_valid(obj, role, board_words):
            return ValidationResult(False, obj, "invalid_codenames_action")
        action = obj["action"]
        if not isinstance(action, dict):
            return ValidationResult(False, obj, "invalid_codenames_action")
        key = "clue" if role == "spymaster" else "guess"
        raw = str(action[key])
        parsed = parse_clue(raw) if role == "spymaster" else parse_guess(raw)
        return ValidationResult(True, parsed)

    if game == "mafia":
        valid_targets = context.get("valid_targets", [])
        if not mafia_action_valid(obj, valid_targets=valid_targets):
            parsed = parse_mafia_action(obj) if isinstance(obj, dict) else None
            reason = "invalid_vote" if parsed else "invalid_format"
            return ValidationResult(False, obj, reason)
        return ValidationResult(True, parse_mafia_action(obj))

    return ValidationResult(False, reason=f"unknown_game:{game}")
