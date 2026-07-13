import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAME = ROOT / "games"


def _load_engine(game: str):
    path = GAME / game / "engine.py"
    spec = importlib.util.spec_from_file_location(f"{game}_engine", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_blotto = _load_engine("blotto")
_ipd = _load_engine("ipd")
_codenames = _load_engine("codenames")
_mafia = _load_engine("mafia")

parse_allocation = _blotto.parse_allocation
is_valid_allocation = _blotto.is_valid_allocation
resolve_round = _blotto.resolve_round
parse_choice = _ipd.parse_choice
simulated_round_reward = _ipd.simulated_round_reward
parse_clue = _codenames.parse_clue
parse_guess = _codenames.parse_guess
is_valid_clue = _codenames.is_valid_clue
parse_mafia_action = _mafia.parse_mafia_action
mafia_action_valid = _mafia.mafia_action_valid


def extract_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def vote_accuracy(pred_vote, gold_vote):
    return int(pred_vote == gold_vote)


def json_validity(obj):
    return int(isinstance(obj, dict) and "action" in obj)


def blotto_json_validity(obj):
    if not isinstance(obj, dict) or "action" not in obj:
        return 0
    action = obj.get("action")
    if not isinstance(action, dict):
        return 0
    alloc = action.get("allocation")
    return int(alloc is not None and parse_allocation(str(alloc)) is not None)


def blotto_allocation_valid(obj) -> int:
    if not blotto_json_validity(obj):
        return 0
    alloc = parse_allocation(str(obj["action"]["allocation"]))
    return int(alloc is not None and is_valid_allocation(alloc))


def blotto_allocation_match(pred_obj, gold_alloc: str) -> int:
    if not blotto_allocation_valid(pred_obj):
        return 0
    pred = parse_allocation(str(pred_obj["action"]["allocation"]))
    gold = parse_allocation(str(gold_alloc))
    return int(pred is not None and gold is not None and pred == gold)


def blotto_round_win(pred_obj, opp_alloc: dict[str, int]) -> int:
    if not blotto_allocation_valid(pred_obj):
        return 0
    pred = parse_allocation(str(pred_obj["action"]["allocation"]))
    if pred is None:
        return 0
    return int(resolve_round(pred, opp_alloc) == "self")


def ipd_json_validity(obj) -> int:
    if not isinstance(obj, dict):
        return 0
    return int("action" in obj)


def ipd_choice_valid(obj) -> int:
    if not isinstance(obj, dict) or "action" not in obj:
        return 0
    action = obj.get("action")
    if not isinstance(action, dict):
        return 0
    choice = action.get("choice")
    return int(parse_choice(str(choice)) in ("cooperate", "defect"))


def ipd_choice_match(pred_obj, gold_choice: str) -> int:
    if not ipd_choice_valid(pred_obj):
        return 0
    pred = parse_choice(str(pred_obj["action"]["choice"]))
    gold = parse_choice(str(gold_choice))
    return int(pred is not None and gold is not None and pred == gold)


def ipd_simulated_reward(pred_obj, joint_actions: dict[str, str], self_id: str) -> int:
    if not ipd_choice_valid(pred_obj):
        return 0
    pred = parse_choice(str(pred_obj["action"]["choice"]))
    if pred is None:
        return 0
    gold_reward = simulated_round_reward(joint_actions[self_id], joint_actions, self_id)
    pred_reward = simulated_round_reward(pred, joint_actions, self_id)
    return int(pred_reward == gold_reward)


def codenames_json_validity(obj, role: str) -> int:
    if not isinstance(obj, dict) or "action" not in obj:
        return 0
    action = obj.get("action")
    if not isinstance(action, dict):
        return 0
    key = "clue" if role == "spymaster" else "guess"
    return int(key in action)


def codenames_action_valid(obj, role: str, board_words: list[str]) -> int:
    if not codenames_json_validity(obj, role):
        return 0
    if role == "spymaster":
        parsed = parse_clue(str(obj["action"]["clue"]))
        if parsed is None:
            return 0
        word, n = parsed
        return int(n >= 1 and is_valid_clue(word, board_words))
    guess = parse_guess(str(obj["action"]["guess"]))
    return int(guess in ("pass",) or guess in [w.lower() for w in board_words])


def codenames_action_match(pred_obj, gold_action: dict, role: str) -> int:
    if not codenames_json_validity(pred_obj, role):
        return 0
    key = "clue" if role == "spymaster" else "guess"
    pred_val = str(pred_obj["action"].get(key, "")).lower().strip()
    gold_val = str(gold_action.get(key, "")).lower().strip()
    if role == "spymaster":
        pred = parse_clue(pred_val)
        gold = parse_clue(gold_val)
        return int(pred is not None and gold is not None and pred == gold)
    return int(parse_guess(pred_val) == parse_guess(gold_val))
