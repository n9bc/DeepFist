"""Text -> ideal keying segments -> click-free on/off envelope."""
from dataclasses import dataclass
import numpy as np

from deepfist.morse.alphabet import SPACE, text_to_tokens, morse_for
from deepfist.morse.timing import Timing


@dataclass
class Segment:
    on: bool
    duration: float  # seconds


def text_to_segments(text: str, timing: Timing) -> list[Segment]:
    segs: list[Segment] = []
    tokens = text_to_tokens(text)
    for i, tok in enumerate(tokens):
        if tok == SPACE:
            segs.append(Segment(False, timing.word_gap))
            continue
        pattern = morse_for(tok)
        for j, elem in enumerate(pattern):
            segs.append(Segment(True, timing.dot if elem == "." else timing.dash))
            if j < len(pattern) - 1:
                segs.append(Segment(False, timing.element_gap))
        # character gap after this token, unless next is a space or end
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and nxt != SPACE:
            segs.append(Segment(False, timing.char_gap))
    return segs


def _bh_step(length: int) -> np.ndarray:
    """Integrated Blackman-Harris kernel -> smooth 0..1 step (click-free ramp)."""
    if length < 1:
        return np.ones(1, dtype=np.float64)
    x = np.arange(length) / length
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    k = a0 - a1*np.cos(2*np.pi*x) + a2*np.cos(4*np.pi*x) - a3*np.cos(6*np.pi*x)
    step = np.cumsum(k)
    return step / step[-1]


def segments_to_envelope(segments: list[Segment], sample_rate: int,
                         rise: float = 0.005) -> np.ndarray:
    total = sum(s.duration for s in segments)
    n = round(total * sample_rate)
    env = np.zeros(n, dtype=np.float64)
    ramp_len = max(1, round(2.7 * rise * sample_rate))
    ramp_on = _bh_step(ramp_len)
    ramp_off = ramp_on[::-1]

    pos = 0
    for s in segments:
        seg_n = round(s.duration * sample_rate)
        if seg_n <= 0:
            continue
        block = np.ones(seg_n) if s.on else np.zeros(seg_n)
        if s.on:
            r = min(ramp_len, seg_n)
            block[:r] = ramp_on[:r]
            block[seg_n - r:] = np.minimum(block[seg_n - r:], ramp_off[ramp_len - r:])
        end = max(0, min(seg_n, n - pos))
        env[pos:pos + end] = block[:end]
        pos += seg_n
    return env.astype(np.float32)
