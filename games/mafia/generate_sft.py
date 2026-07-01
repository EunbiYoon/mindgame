#!/usr/bin/env python3
import argparse, json, random

ROLES = ["Mafia", "Detective", "Doctor", "Citizen", "Citizen"]
PLAYERS = ["A", "B", "C", "D", "E"]

MAFIA_LINES = [
    "I think {target} is pushing too hard. That feels suspicious.",
    "{target}'s logic does not add up to me.",
    "I am citizen. We should not blindly trust claims.",
]
CITIZEN_LINES = [
    "I want to compare vote history before deciding.",
    "{target} changed their story, so I am suspicious.",
    "A fake claim could be happening here.",
]
DETECTIVE_LINES = [
    "I checked {target}. My read is that {target} is suspicious.",
    "Based on my night check, {target} should be pressured.",
]
DOCTOR_LINES = [
    "I do not want to reveal too much, but we should protect useful claims.",
    "{target} is attracting attention; mafia may exploit that.",
]


def assign_roles(rng):
    roles = ROLES[:]
    rng.shuffle(roles)
    return dict(zip(PLAYERS, roles))


def make_chat(player, role, target, rng):
    if role == "Mafia":
        return rng.choice(MAFIA_LINES).format(target=target)
    if role == "Detective":
        return rng.choice(DETECTIVE_LINES).format(target=target)
    if role == "Doctor":
        return rng.choice(DOCTOR_LINES).format(target=target)
    return rng.choice(CITIZEN_LINES).format(target=target)


def choose_target(player, roles, rng):
    others = [p for p in PLAYERS if p != player]
    mafia = [p for p in others if roles[p] == "Mafia"]
    if roles[player] != "Mafia" and mafia and rng.random() < 0.65:
        return mafia[0]
    non_mafia = [p for p in others if roles[p] != "Mafia"]
    return rng.choice(non_mafia or others)


def make_example(roles, self_id, rng, day):
    alive = PLAYERS[:]
    mafia = [p for p, r in roles.items() if r == "Mafia"][0]
    chat = []
    claims = {}
    for p in alive:
        target = choose_target(p, roles, rng)
        text = make_chat(p, roles[p], target, rng)
        chat.append({"speaker": p, "text": text})
        if roles[p] == "Detective" and rng.random() < 0.7:
            claims[p] = "Detective"
        elif roles[p] == "Mafia" and rng.random() < 0.25:
            claims[p] = rng.choice(["Citizen", "Detective"])
        else:
            claims[p] = "Citizen"

    votes = {p: choose_target(p, roles, rng) for p in alive}
    belief = {
        p: {"mafia": 0.85 if p == mafia else round(rng.uniform(0.05, 0.35), 2)}
        for p in alive
        if p != self_id
    }
    vote = mafia if roles[self_id] != "Mafia" else rng.choice([p for p in alive if roles[p] != "Mafia"])
    reasoning = (
        f"The strongest hidden-role hypothesis is that {mafia} is mafia. "
        "This follows from accusations, claim consistency, and vote pressure."
        if roles[self_id] != "Mafia"
        else "As mafia, I should redirect suspicion toward a non-mafia player while sounding cautious."
    )
    speak = (
        f"I think we should pressure {vote}. Their statement and voting pattern look inconsistent."
        if roles[self_id] != "Mafia"
        else f"I am not convinced by the case on me. {vote} looks more inconsistent."
    )

    inp = {
        "game_state": {
            "day": day,
            "alive": alive,
            "dead": [],
            "public_chat": chat,
            "vote_history": [{"day": max(1, day - 1), "votes": votes}],
            "claims": claims,
        },
        "private_info": {"self_id": self_id, "self_role": roles[self_id]},
        "task": "Infer likely mafia and choose the next speech and vote.",
    }
    out = {"belief": belief, "reasoning": reasoning, "action": {"speak": speak, "vote": vote}}
    prompt = (
        "You are playing Secret Mafia. Use opponent modeling and theory of mind.\n"
        "Given the state, infer hidden roles and choose speech/vote.\n\n"
        f"STATE:\n{json.dumps(inp, ensure_ascii=False)}\n\n"
        "Respond as JSON with keys: belief, reasoning, action."
    )
    return {
        "prompt": prompt,
        "completion": json.dumps(out, ensure_ascii=False),
        "input": inp,
        "output": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="games/mafia/sft.jsonl")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n_games):
            roles = assign_roles(rng)
            for self_id in PLAYERS:
                ex = make_example(roles, self_id, rng, day=rng.randint(1, 3))
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n_games * len(PLAYERS)} examples to {args.out}")


if __name__ == "__main__":
    main()
