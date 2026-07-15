"""CW ragchew language model — STUB (not yet wired into decode).

Purpose (see HANDOFF §18.23 "predictive common words"): a CW-domain n-gram LM to
gently bias decoding toward likely QSO phraseology, fused with the model's CTC
acoustic scores in a beam search (shallow fusion) — NOT a blind post-hoc word
snap. The acoustic evidence must be able to override the LM so rare-but-critical
tokens (callsigns, grids, park numbers, reports) survive.

DO NOT ENABLE until base acoustic copy is decent: an LM over garbled copy invents
fluent QSO phrases that were never sent — worse than garble, and a direct
violation of the "don't make things up" rule. Gate on real ARRL CER + real eval.

Integration plan (later):
  - Train n-gram from CW QSO/ragchew text (ARRL practice text, RBN logs, W1AW
    bulletins, ham QSO corpora).
  - Add a beam-search CTC decoder (deepfist.model.decode) that adds
    lm_weight * lm.logprob(prefix+word) to the acoustic score per word.
  - PROTECT rare tokens: never let the LM rewrite a token matching RARE_RE
    (callsign / number / grid / Q-code shapes) — the C5POTA/callsign-snap trap.
  - Surface LM-assisted words distinctly (mark), never silent-replace.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import log

# Tokens the LM must NEVER "correct" toward a common word (acoustics wins here).
RARE_RE = re.compile(
    r"^(?:[A-Z0-9]{1,3}/)?[A-Z]{1,2}\d{1,3}[A-Z]{1,4}(?:/[A-Z0-9]{1,3})?$"  # callsign
    r"|^\d+$"                                                                 # numbers
    r"|^[A-R]{2}\d{2}(?:[A-X]{2})?$"                                          # Maidenhead grid
)

# Seed CW ragchew vocabulary / abbreviations (frequency-weighted later from corpus).
CW_COMMON = [
    "CQ", "DE", "K", "KN", "R", "RR", "BK", "AR", "SK", "AS",
    "UR", "RST", "URS", "ES", "PSE", "TNX", "TU", "FB", "HR", "HW",
    "CPY", "AGN", "GM", "GA", "GE", "GN", "OM", "YL", "XYL", "DR",
    "NAME", "QTH", "RIG", "ANT", "PWR", "WX", "TEMP", "HPE", "CUL",
    "73", "88", "599", "5NN", "QSL", "QRZ", "QRM", "QRN", "QSB", "QRP",
    "QSO", "QTC", "QRL", "QSY", "WID", "GUD", "GD", "VY", "NW", "BTU",
    "THE", "AND", "IS", "IN", "ON", "AT", "TO", "OF", "FOR", "HERE",
    "POTA", "SOTA", "PARK", "TEST", "CONTEST",
]


class CwLM:
    """Minimal word bigram over CW text. STUB — build/score work; not integrated."""

    def __init__(self, order: int = 2):
        self.order = order
        self.uni: Counter = Counter()
        self.bi: dict[str, Counter] = defaultdict(Counter)
        self._total = 0
        for t in CW_COMMON:                     # weak seed prior
            self.uni[t] += 1
            self._total += 1

    def train(self, texts) -> "CwLM":
        for line in texts:
            toks = [t for t in re.split(r"\s+", line.upper().strip()) if t]
            prev = "<s>"
            for t in toks:
                self.uni[t] += 1
                self.bi[prev][t] += 1
                self._total += 1
                prev = t
        return self

    def logprob(self, word: str, prev: str | None = None) -> float:
        """Add-1 smoothed log P(word|prev). Placeholder scoring for fusion."""
        V = max(1, len(self.uni))
        if prev and self.bi.get(prev):
            c = self.bi[prev]
            return log((c[word] + 1) / (sum(c.values()) + V))
        return log((self.uni[word] + 1) / (self._total + V))

    def is_protected(self, token: str) -> bool:
        """True if the LM must not rewrite this token (callsign/number/grid)."""
        return bool(RARE_RE.match(token.upper()))


def load_default() -> CwLM:
    """Seed-only LM (no corpus yet). Replace with a corpus-trained model later."""
    return CwLM()


if __name__ == "__main__":
    lm = load_default()
    for w, p in [("QTH", "MY"), ("RST", "UR"), ("WEATHER", None), ("KE9XYZ", None)]:
        print(f"{w:8} prev={p!s:5}  logP={lm.logprob(w, p):.2f}  "
              f"protected={lm.is_protected(w)}")
