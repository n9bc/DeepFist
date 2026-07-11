"""Canonical Morse alphabet: unique-pattern tokens, tokenization, prosigns."""
import re

BLANK = "<blank>"
SPACE = " "

_LETTERS = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
}
_DIGITS = {
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..", "9": "----.", "0": "-----",
}
_PUNCT = {
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.",
    "=": "-...-", "+": ".-.-.", "-": "-....-", "@": ".--.-.",
}
_PROSIGNS = {"<SK>": "...-.-", "<KN>": "-.--."}

# Prosigns that are acoustically identical to a punctuation token collapse to it.
ALIASES = {"<AR>": "+", "<BT>": "="}

MORSE: dict[str, str] = {**_LETTERS, **_DIGITS, **_PUNCT, **_PROSIGNS}

TOKENS: list[str] = (
    [BLANK, SPACE]
    + list(_LETTERS) + list(_DIGITS) + list(_PUNCT) + list(_PROSIGNS)
)
TOKEN_TO_ID: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}

_TOKEN_RE = re.compile(r"<[A-Z]+>|\s+|.", re.IGNORECASE)


def morse_for(token: str) -> str:
    if token == SPACE:
        return ""
    return MORSE[token]  # KeyError for BLANK / unknown


def text_to_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        piece = m.group(0)
        if piece.isspace():
            if tokens and tokens[-1] == SPACE:
                continue
            tokens.append(SPACE)
        elif piece.startswith("<"):
            up = piece.upper()
            if up in ALIASES:
                tokens.append(ALIASES[up])
            elif up in _PROSIGNS:
                tokens.append(up)
            else:
                raise ValueError(f"Unknown prosign: {piece}")
        else:
            up = piece.upper()
            if up not in MORSE:
                raise ValueError(f"Unknown symbol: {piece!r}")
            tokens.append(up)
    return tokens


def tokens_to_text(tokens: list[str]) -> str:
    return "".join(tokens)
