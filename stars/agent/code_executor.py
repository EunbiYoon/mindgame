"""Safe Python execution for STARS ReAct / PAL steps."""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from typing import Any

from stars.agent.belief_state import BeliefState


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def execute_react_code(code: str, belief: BeliefState) -> dict[str, Any]:
    """Run agent-generated Python against the belief state namespace."""
    if not code or not code.strip():
        return {"stdout": "", "error": "empty code", "result": {}}

    namespace: dict[str, Any] = {
        "belief": belief,
        "state": belief.to_context(),
        "alive_players": list(belief.alive_players),
        "vote_history": list(belief.vote_history),
        "p_mafia": dict(belief.p_mafia),
        "trust_score": dict(belief.trust_score),
        "claims": dict(belief.claims),
        "result": {},
    }
    stdout = io.StringIO()
    error = ""
    try:
        with redirect_stdout(stdout):
            exec(code, {"__builtins__": _SAFE_BUILTINS}, namespace)
    except Exception:
        error = traceback.format_exc(limit=3)

    result = namespace.get("result", {})
    if not isinstance(result, dict):
        result = {"value": result}

    # Allow code to mutate copies back into belief via `result` dict.
    belief.apply_code_result(result)

    return {
        "stdout": stdout.getvalue().strip(),
        "error": error,
        "result": result,
        "p_mafia": dict(belief.p_mafia),
        "trust_score": dict(belief.trust_score),
    }


def validation_code(action_vote: str, valid_targets: list[str], alive: list[str]) -> dict[str, Any]:
    vote = str(action_vote).strip()
    valid = set(valid_targets) or set(alive)
    return {
        "valid": vote in valid and vote != "",
        "vote": vote,
        "valid_targets": sorted(valid),
    }
