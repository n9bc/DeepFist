"""Decode-time QSO-shorthand corrector — no retraining.

exp16 decodes real CW at ~3% char error; on rag-chew content that garbles known operating
shorthand (TNX->TXN, QTH->QTX). Snap each decoded WORD to the nearest known CW abbreviation/
Q-code within edit-distance 1 -- but CONSERVATIVELY (over-correction hurt the callsign path):
only short words, only a 1-char fix, never touch callsign- or number-shaped tokens. Must
help ham shorthand WITHOUT damaging plain English prose (validated on real_arrl_eval).
"""
from __future__ import annotations
import re

# CW operating vocabulary (abbreviations, Q-codes, prosigns) from ham CW references.
VOCAB = {
    "ABT", "AGN", "ANS", "ANT", "BK", "BN", "BTW", "CFM", "CL", "CLG", "CONDX", "CPY",
    "CQ", "CW", "DE", "DR", "DX", "ES", "FB", "FER", "GA", "GB", "GD", "GE", "GM", "GN",
    "GND", "GP", "GUD", "HAM", "HI", "HR", "HV", "HW", "K", "KN", "LID", "MNI", "MSG",
    "NIL", "NR", "NW", "OM", "OP", "PSE", "PWR", "R", "RCVD", "RIG", "RPT", "RPRT", "RST",
    "RX", "SIG", "SK", "SKED", "SN", "SRI", "STN", "TEST", "TFC", "TIL", "TKS", "TNX",
    "TU", "TX", "UFB", "UR", "VY", "WKD", "WPM", "WX", "YL", "XYL",
    "QRZ", "QTH", "QSL", "QSO", "QSY", "QRM", "QRN", "QSB", "QRP", "QRT", "QRL", "QRX",
    "IS", "AND", "THE", "NAME", "HERE", "HOW", "GUD", "MORNING", "GOOD",
}
_CALL = re.compile(r"^[A-Z]{1,2}[0-9][A-Z]{1,4}(/[0-9A-Z]+)?$")
_NUM = re.compile(r"^[0-9]+$")
# length-bucketed vocab for fast ED-1 lookup
_BY_LEN: dict[int, list[str]] = {}
for _w in VOCAB:
    _BY_LEN.setdefault(len(_w), []).append(_w)


def _ed1(a: str, b: str) -> bool:
    """True if edit distance(a,b) <= 1 (same or one substitution/insert/delete)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                                   # one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b = b, a                                # ensure a shorter
    for i in range(len(b)):                        # one deletion from b
        if a == b[:i] + b[i + 1:]:
            return True
    return False


def correct_word(w: str) -> str:
    if not w or w in VOCAB or _CALL.match(w) or _NUM.match(w):
        return w                                   # keep valid/callsign/number as-is
    if not (2 <= len(w) <= 6):
        return w                                   # only short words are shorthand
    for L in (len(w), len(w) - 1, len(w) + 1):
        for cand in _BY_LEN.get(L, ()):
            if _ed1(w, cand):
                return cand
    return w


def correct_qso(text: str) -> str:
    """Snap each whitespace-token to the nearest CW vocab word (ED<=1, conservative)."""
    return " ".join(correct_word(t) for t in text.split())


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    import numpy as np, torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
    from deepfist.features.conditioner import maybe_condition
    from deepfist.model.decode import greedy_ctc_decode
    from deepfist.train.metrics import cer
    from deepfist.synth.generator import generate, GenConfig
    import callsign_eval as CE
    net = CE._load_net("runs/exp16/model.pt")

    def dec_arr(a, sr):
        if a.ndim > 1: a = a.mean(1)
        a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
        if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
        with torch.no_grad():
            lp = net(audio_to_spectrogram(maybe_condition(a, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
        return greedy_ctc_decode(lp)[0]

    # 1) SAFETY: plain English prose (real ARRL) must not get worse
    D = ROOT / "runs" / "real_arrl_eval"
    rows = [json.loads(l) for l in (D / "labels.jsonl").read_text().splitlines() if l.strip()]
    raw = corr = 0.0
    for r in rows:
        sr, a = wavfile.read(str(D / r["file"])); p = dec_arr(a, sr)
        raw += cer(p, r["text"]); corr += cer(correct_qso(p), r["text"])
    n = len(rows)
    print(f"ARRL prose (safety): raw CER {raw/n:.3f} -> corrected {corr/n:.3f}  ({n} clips)")

    # 2) TARGET: synthetic rag-chew (known text) should improve
    cfg = GenConfig(snr_range=(0.0, 15.0))
    raw = corr = 0.0; n = 0
    seed = 500000
    while n < 120:
        s = generate(seed=seed, config=cfg); seed += 1
        if "=" not in s.label and not any(t in s.label.split() for t in ("TNX", "UR", "RST", "QTH", "DE")):
            continue                               # keep rag-chew-style utterances
        p = dec_arr(np.asarray(s.audio, np.float32), cfg.sample_rate)
        raw += cer(p, s.label); corr += cer(correct_qso(p), s.label); n += 1
    print(f"rag-chew synth (target): raw CER {raw/n:.3f} -> corrected {corr/n:.3f}  ({n} clips)")
