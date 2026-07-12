"""Contest CW exchange generator.

Targets the contests DeepFist is scoped to: CQ WW (zone), CQ WPX (serial),
ARRL DX (state/province or power), and state QSO parties (state). Emits single
realistic on-air utterances (a CQ, an answer, an exchange, a confirmation, or a
repeat request) rather than random word salad, so the model trains on the actual
structure it will hear — and can lean on that grammar to error-correct in noise.
"""
import numpy as np

from deepfist.morse.alphabet import text_to_tokens

# Callsign building blocks (1- and 2-char prefixes across common contest regions)
_PFX1 = ["K", "W", "N", "G", "F", "I", "R"]
_PFX2 = ["AA", "AB", "KA", "KB", "KC", "KE", "WA", "WB", "NA", "VE", "VA", "DL",
         "DF", "EA", "EA", "GM", "GW", "ON", "PA", "SP", "OK", "OM", "OH", "OZ",
         "LZ", "YO", "UA", "UR", "JA", "JH", "VK", "ZL", "PY", "LU", "CE", "HA"]
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# US states + Canadian provinces (ARRL DX from W/VE side, QSO parties)
_STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL",
           "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
           "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
           "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
           "WY", "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
_WATTS = [5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 800, 900]


def _cut(text, rng, p=0.5):
    """Contest cut numbers on an exchange NUMBER: 0->T (rarely O), 9->N (1->A rare).
    Never applied to callsigns."""
    out = []
    for ch in text:
        if ch == "0" and rng.random() < p:
            out.append("T" if rng.random() < 0.85 else "O")
        elif ch == "9" and rng.random() < p:
            out.append("N")
        elif ch == "1" and rng.random() < p * 0.25:
            out.append("A")
        else:
            out.append(ch)
    return "".join(out)


def _power(rng):
    """ARRL DX power exchange: KW/K (1000W), QRP (<=5W), or watts with cut zeros
    (100->1TT, 500->5TT)."""
    r = rng.random()
    if r < 0.45:
        return "K" if rng.random() < 0.5 else "KW"
    if r < 0.58:
        return "QRP"
    w = int(rng.choice(_WATTS))
    return "".join("T" if c == "0" else "N" if c == "9" else c for c in str(w))


def _callsign(rng):
    if rng.random() < 0.45:
        pfx = str(rng.choice(_PFX1))
    else:
        pfx = str(rng.choice(_PFX2))
    digit = str(rng.integers(0, 10))
    n = int(rng.choice([1, 2, 2, 3]))          # 1x1..2x3, weighted to 2-letter suffix
    tail = "".join(rng.choice(list(_LETTERS), size=n))
    call = f"{pfx}{digit}{tail}"
    if rng.random() < 0.08:                     # portable
        call += "/" + str(rng.integers(0, 10))
    return call


def _report(rng):
    return "5NN" if rng.random() < 0.75 else "599"


def _serial(rng):
    n = int(rng.integers(1, 3000))
    s = f"{n:03d}" if n < 1000 and rng.random() < 0.8 else str(n)   # early ones zero-padded
    return _cut(s, rng)


def _zone(rng):
    return _cut(f"{int(rng.integers(1, 41)):02d}", rng)


def _exchange(rng):
    """Pick a contest and return the post-report exchange string."""
    r = rng.random()
    if r < 0.35:                 # CQ WPX
        return _serial(rng)
    if r < 0.60:                 # CQ WW
        return _zone(rng)
    if r < 0.85:                 # ARRL DX / QSO party -> state/province
        return str(rng.choice(_STATES))
    return _power(rng)           # ARRL DX from DX side -> power


def random_message(rng: np.random.Generator, max_tokens: int = 40) -> str:
    """One realistic contest utterance."""
    call = _callsign(rng)
    roll = rng.random()

    if roll < 0.22:              # CQ (running station)
        me = _callsign(rng)
        msg = rng.choice([f"CQ TEST {me} {me}", f"CQ TEST {me}",
                          f"CQ CQ TEST {me}", f"TEST {me} {me}", f"CQ {me} TEST"])
    elif roll < 0.40:            # answer: just the caller's callsign
        msg = call if rng.random() < 0.6 else f"{call} {call}"
    elif roll < 0.72:            # exchange (report + contest exchange)
        exch = _exchange(rng)
        rep = _report(rng)
        msg = rng.choice([f"{call} {rep} {exch}", f"{rep} {exch}",
                          f"{call} {rep} {exch} {exch}", f"{rep} {rep} {exch}"])
    elif roll < 0.88:            # confirmation
        me = _callsign(rng)
        msg = rng.choice([f"TU {me}", f"R {_report(rng)} {_exchange(rng)} TU",
                          f"TU {me} TEST", f"R TU", f"{call} TU"])
    else:                        # repeat / fill request
        msg = rng.choice([f"{call} ?", "NR?", "AGN", "?", f"{call} AGN",
                          "QRZ?", f"NR {_exchange(rng)}?"])

    msg = str(msg)
    toks = text_to_tokens(msg)
    while len(toks) > max_tokens and " " in msg:
        msg = msg.rsplit(" ", 1)[0]
        toks = text_to_tokens(msg)
    return msg or "TEST"
