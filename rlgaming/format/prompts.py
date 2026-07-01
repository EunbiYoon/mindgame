"""Formatting-only prompts for RLGaming (strategy comes from SFT, not prompt)."""

FORMAT_SYSTEM = {
    "mafia": (
        "You are playing Secret Mafia. "
        "Use only the information in the state blocks below. "
        "Respond in the format required by the current phase."
    ),
    "blotto": (
        "You are Commander in Colonel Blotto. "
        "Use only the information in the state blocks below. "
        "Allocate units as [A# B# C#] summing to the available units."
    ),
    "ipd": (
        "You are playing Three-Player IPD. "
        "Use only the information in the state blocks below. "
        "Match the current phase format (chat or decision tokens)."
    ),
    "codenames": (
        "You are playing Codenames. "
        "Use only the information in the state blocks below. "
        "Spymaster: [word n]. Operative: [word] or [pass]."
    ),
}

FORMAT_PROMPT = """{context}

ANSWER in the format required by the current phase. Do not add extra commentary unless the phase expects discussion."""
