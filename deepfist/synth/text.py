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


# --- rag-chew (conversational QSO) vocabulary ------------------------------------
_NAMES = ["JOHN", "BOB", "JIM", "TOM", "MIKE", "DAVE", "BILL", "STEVE", "DAN", "RON",
          "GARY", "KEN", "ED", "FRANK", "PAUL", "JACK", "LARRY", "CARL", "HANK", "AL",
          "PETE", "RAY", "DON", "JOE", "RICK", "WALT", "CHUCK", "GENE", "ART", "LOU"]
_CITIES = ["BOSTON", "NEW YORK", "CHICAGO", "DENVER", "DALLAS", "SEATTLE", "MIAMI",
           "ATLANTA", "PHOENIX", "AUSTIN", "PORTLAND", "TAMPA", "RENO", "OMAHA",
           "TULSA", "AKRON", "FARGO", "BOISE", "SALEM", "TROY", "MESA", "ERIE"]
_RIGS = ["FT991", "IC7300", "K3", "TS590", "KX3", "FLEX", "IC7610", "FT817", "TS890",
         "K4", "FTDX10", "IC705", "HOMEBREW", "HB", "FT101", "TS520", "ARGO"]
_ANTS = ["DIPOLE", "VERTICAL", "YAGI", "GP", "LW", "EFHW", "LOOP", "G5RV", "BEAM",
         "INV V", "HEXBEAM", "MONOBANDER", "WINDOM", "SLOPER", "END FED", "40M DIPOLE"]
_WX = ["SUNNY", "RAIN", "CLOUDY", "SNOW", "CLEAR", "WINDY", "COLD", "HOT", "FOG",
       "WARM", "COOL", "NICE WX", "RAINY", "MILD", "HAZY"]
_GREET = ["GM", "GA", "GE", "GM DR OM", "GE DR OM", "GA OM", "GM OM", "GE OM", "GD"]
_RST = ["599", "579", "589", "559", "449", "569", "578", "339", "588", "459"]


def _rst(rng):
    return _cut(rng.choice(_RST), rng) if rng.random() < 0.4 else str(rng.choice(_RST))


def _ragchew(rng):
    """One over of a conversational CW QSO, built from real operating shorthand."""
    call, me = _callsign(rng), _callsign(rng)
    nm = str(rng.choice(_NAMES))
    r = rng.random()
    if r < 0.15:                       # CQ / calling
        return rng.choice([f"CQ CQ DE {me} {me} K", f"CQ CQ CQ DE {me} {me} {me} K",
                           f"CQ DE {me} {me} K", f"QRZ DE {me} K"])
    if r < 0.28:                       # answer
        return rng.choice([f"{call} DE {me} {me} K", f"{call} DE {me} = GE = KN",
                           f"{me} DE {call} K"])
    if r < 0.75:                       # info exchange (the meat of a rag-chew)
        bits = [str(rng.choice(_GREET))]
        if rng.random() < 0.7:
            bits.append(rng.choice([f"TNX FER CALL", f"TKS FER QSO", f"MNI TNX FER RPRT",
                                    f"FB {nm}", f"UFB SIG"]))
        if rng.random() < 0.85:
            bits.append(f"UR RST {_rst(rng)} {_rst(rng)}")
        if rng.random() < 0.8:
            bits.append(rng.choice([f"QTH {rng.choice(_CITIES)}", f"QTH IS {rng.choice(_CITIES)}",
                                    f"HR IN {rng.choice(_CITIES)}"]))
        if rng.random() < 0.8:
            bits.append(rng.choice([f"NAME {nm}", f"NAME IS {nm}", f"OP {nm} {nm}"]))
        if rng.random() < 0.5:
            bits.append(rng.choice([f"RIG {rng.choice(_RIGS)} ES ANT {rng.choice(_ANTS)}",
                                    f"RIG IS {rng.choice(_RIGS)}", f"ANT {rng.choice(_ANTS)}",
                                    f"PWR {int(rng.integers(5, 100))}W"]))
        if rng.random() < 0.4:
            bits.append(rng.choice([f"WX {rng.choice(_WX)}", f"WX IS {rng.choice(_WX)} HR"]))
        body = " = ".join(bits)
        return rng.choice([f"{call} DE {me} = {body} = HW? {call} DE {me} KN",
                           f"{call} DE {me} = {body} = BK",
                           f"R R = {body} = HW CPY? {call} DE {me} KN"])
    if r < 0.9:                        # sign-off
        return rng.choice([f"{call} DE {me} = TNX FER FB QSO {nm} = HPE CUAGN = 73 ES GB = {call} DE {me} SK",
                           f"{call} DE {me} = MNI TNX = 73 73 = {me} SK",
                           f"R FB {nm} = TNX QSO = 73 ES CUL = {call} DE {me} SK",
                           f"{call} DE {me} = GB {nm} = 73 = SK"])
    # short acks / fills
    return rng.choice([f"{call} DE {me} = R R = SRI QRM = PSE RPT UR NAME = KN",
                       f"AGN? PSE AGN = {call} DE {me} KN", f"R R FB = QSL = BK",
                       f"NIL = PSE RPT = KN", f"{me} = QRZ? = K"])


# --- BT prosign discrimination drill (exp27) ------------------------------------
# exp16 misreads BT (=, -...-) as X (-..-) / 6 (-....) / B (-...) at ALL speeds,
# on perfect synthetic AND perfect real hand-sent BT (operator N9BC, verified) —
# a genuine discrimination weakness on the 3rd dit + final dah, NOT a coverage gap
# (BT is already in 37% of messages). These drills put BT directly next to its
# dit-heavy look-alikes so the CTC loss is FORCED to resolve them (incidental BT
# buried in words carries too little relative loss to teach the distinction), and
# also cover the double "==" the operator sent (0% in the base distribution).
_BT_LOOKALIKE = ["X", "6", "B", "5", "H", "S", "4", "D", "N", "T", "G", "W"]


def _prosign_drill(rng: np.random.Generator) -> str:
    if rng.random() < 0.5:
        # Direct discrimination: BT interleaved with its look-alikes.
        k = int(rng.integers(5, 10))
        toks = ["=" if rng.random() < 0.45 else str(rng.choice(_BT_LOOKALIKE))
                for _ in range(k)]
        if "=" not in toks:
            toks[int(rng.integers(0, k))] = "="
        return " ".join(toks)
    # Realistic BT-heavy usage, incl. the double "==" separator.
    call, me = _callsign(rng), _callsign(rng)
    return str(rng.choice([
        "R = R = TU = 73 =",
        f"{call} = = {me}",
        "= = 5NN = =",
        f"BK = {call} = HW? =",
        "TU = 73 = = GB",
        f"{me} = X 6 = B = 6 X =",
    ]))


def random_message(rng: np.random.Generator, max_tokens: int = 40) -> str:
    """One realistic on-air utterance: ~half contest, ~half conversational rag-chew."""
    if rng.random() < 0.10:                 # exp27: BT discrimination drill
        msg = _prosign_drill(rng)
        toks = text_to_tokens(msg)
        while len(toks) > max_tokens and " " in msg:
            msg = msg.rsplit(" ", 1)[0]
            toks = text_to_tokens(msg)
        return msg or "TEST"
    if rng.random() < 0.5:
        msg = _ragchew(rng)
        toks = text_to_tokens(msg)
        while len(toks) > max_tokens and " " in msg:
            msg = msg.rsplit(" ", 1)[0]
            toks = text_to_tokens(msg)
        return msg or "TEST"
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
