"""Simple non-ML envelope decoder — validates the generator on clean audio only."""
import numpy as np
from deepfist.morse.alphabet import MORSE

_PATTERN_TO_CHAR = {v: k for k, v in MORSE.items()}


def _envelope(audio, sample_rate, pitch_hz=None):
    """Keying envelope. When the carrier pitch is known, use coherent I/Q
    demodulation (ripple-free, pitch-independent); otherwise fall back to
    rectify-and-smooth."""
    x = audio.astype(np.float64)
    win = max(1, int(0.004 * sample_rate))
    kernel = np.ones(win) / win
    if pitch_hz:
        t = np.arange(len(x)) / sample_rate
        i = np.convolve(x * np.cos(2 * np.pi * pitch_hz * t), kernel, mode="same")
        q = np.convolve(x * np.sin(2 * np.pi * pitch_hz * t), kernel, mode="same")
        return 2.0 * np.sqrt(i * i + q * q)
    return np.convolve(np.abs(x), kernel, mode="same")


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
    env = _envelope(audio, sample_rate, pitch_hz)
    thr = 0.25 * env.max()
    on = env > thr
    unit = (1.2 / wpm) * sample_rate  # samples per dot (nominal)
    runs = list(_runs(on))

    # Adaptive dit/dah boundary (fldigi-style): fist jitter + envelope smoothing
    # shift ON durations off nominal, so a fixed 2*unit split misreads shortened
    # dahs as dits at high wpm. When both clusters are present, split at the
    # midpoint of the measured dit/dah means instead.
    on_split = 2 * unit
    on_lens = np.asarray([ln for v, ln in runs if v], dtype=float)
    if len(on_lens) >= 2 and on_lens.max() / max(on_lens.min(), 1.0) > 1.8:
        mid = (on_lens.min() + on_lens.max()) / 2
        dits, dahs = on_lens[on_lens <= mid], on_lens[on_lens > mid]
        if len(dits) and len(dahs):
            on_split = (dits.mean() + dahs.mean()) / 2

    text, pattern = [], ""

    def flush():
        nonlocal pattern
        if pattern:
            text.append(_PATTERN_TO_CHAR.get(pattern, "?"))
            pattern = ""

    for value, length in runs:
        if value:
            pattern += "." if length < on_split else "-"
        else:
            u = length / unit
            if u >= 5:      # word gap
                flush(); text.append(" ")
            elif u >= 2:    # char gap
                flush()
            # else: element gap -> keep building the character
    flush()
    return "".join(text).strip()
