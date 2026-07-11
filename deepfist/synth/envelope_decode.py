"""Simple non-ML envelope decoder — validates the generator on clean audio only."""
import numpy as np
from deepfist.morse.alphabet import MORSE

_PATTERN_TO_CHAR = {v: k for k, v in MORSE.items()}


def _envelope(audio, sample_rate):
    rect = np.abs(audio.astype(np.float64))
    win = max(1, int(0.003 * sample_rate))
    kernel = np.ones(win) / win
    return np.convolve(rect, kernel, mode="same")


def _runs(mask):
    """Yield (value, length) run-length pairs over a boolean array."""
    if len(mask) == 0:
        return
    idx = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], idx))
    ends = np.concatenate((idx, [len(mask)]))
    for s, e in zip(starts, ends):
        yield bool(mask[s]), e - s


def decode_clean_envelope(audio, sample_rate, pitch_hz, wpm) -> str:
    env = _envelope(audio, sample_rate)
    thr = 0.25 * env.max()
    on = env > thr
    unit = (1.2 / wpm) * sample_rate  # samples per dot
    text, pattern = [], ""

    def flush():
        nonlocal pattern
        if pattern:
            text.append(_PATTERN_TO_CHAR.get(pattern, "?"))
            pattern = ""

    for value, length in _runs(on):
        u = length / unit
        if value:
            pattern += "." if u < 2 else "-"
        else:
            if u >= 5:      # word gap
                flush(); text.append(" ")
            elif u >= 2:    # char gap
                flush()
            # else: element gap -> keep building the character
    flush()
    return "".join(text).strip()
