"""Random realistic CW message generator (callsigns, QSO fragments, words)."""
import numpy as np
from deepfist.morse.alphabet import text_to_tokens

_PREFIXES = ["W", "K", "N", "G", "DL", "F", "EA", "VE", "VK", "JA", "PA", "OH"]
_SUFFIXES = ["CQ", "TEST", "DE", "K", "KN", "TU", "73", "GM", "GE", "5NN", "599", "R"]
_WORDS = ["THE", "AND", "FB", "UR", "RST", "NAME", "QTH", "WX", "RIG", "ANT", "PWR", "HR"]
_DIGITS = "0123456789"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _callsign(rng: np.random.Generator) -> str:
    pfx = str(rng.choice(_PREFIXES))
    digit = str(rng.choice(list(_DIGITS)))
    n = int(rng.integers(1, 4))
    tail = "".join(rng.choice(list(_LETTERS), size=n))
    call = f"{pfx}{digit}{tail}"
    if rng.random() < 0.15:  # portable
        call += "/" + str(rng.choice(list(_DIGITS)))
    return call


def random_message(rng: np.random.Generator, max_tokens: int = 40) -> str:
    parts: list[str] = []
    n_parts = int(rng.integers(2, 7))
    for _ in range(n_parts):
        roll = rng.random()
        if roll < 0.4:
            parts.append(_callsign(rng))
        elif roll < 0.7:
            parts.append(str(rng.choice(_SUFFIXES)))
        else:
            parts.append(str(rng.choice(_WORDS)))
    msg = " ".join(parts)
    # Trim (whole tokens) to fit max_tokens.
    while len(text_to_tokens(msg)) > max_tokens and " " in msg:
        msg = msg.rsplit(" ", 1)[0]
    toks = text_to_tokens(msg)
    if len(toks) > max_tokens:
        msg = msg[:max_tokens]
    return msg or "CQ"
