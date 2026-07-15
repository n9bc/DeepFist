"""Impulse noise blanker (de-spike) — runs on model-rate audio BEFORE conditioning.

Borrowed from fldigi's CW decoder, which rejects any element shorter than dit/2
as noise. Static crashes / key-clicks are brief BROADBAND spikes; a real CW tone
is a SUSTAINED narrowband burst (dit ~60 ms @ 20 wpm). So an isolated excursion
that towers over the locally-sustained level (median-filtered envelope) is an
impulse, not signal — blank it. During key-down the local median is already the
tone level, so a real dit won't exceed k x its own median; only a true spike does.

Neutral on clean audio, helps on crashy/noisy signals. Validate on real ARRL CER
+ real captures (synthetic-only gains are not trusted here).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, maximum_filter


def despike(x: np.ndarray, sr: int, k: float = 5.0,
            win_ms: float = 8.0, guard_ms: float = 0.5) -> np.ndarray:
    """Blank brief impulse spikes. Returns audio same length/scale as input.

    k        : an envelope sample is a spike if it exceeds k x the local median.
    win_ms   : window for the median baseline (~a couple dit-fractions).
    guard_ms : also blank this much on each side of a detected spike.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    n = x.size
    if n < 8:
        return x
    ax = np.abs(x)
    w = max(3, int(sr * win_ms / 1000) | 1)          # odd window
    base = median_filter(ax, size=w) + 1e-6
    spike = ax > (k * base)
    if not spike.any():
        return x
    g = max(0, int(sr * guard_ms / 1000))
    if g:
        spike = maximum_filter(spike, size=2 * g + 1)  # dilate by guard
    y = x.copy()
    y[spike] = 0.0
    return y


def spike_fraction(x: np.ndarray, sr: int, **kw) -> float:
    """Fraction of samples flagged as impulse — a quick 'how crashy' meter."""
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x.mean(1) if x.ndim > 1 else x)
    w = max(3, int(sr * kw.get("win_ms", 8.0) / 1000) | 1)
    base = median_filter(ax, size=w) + 1e-6
    return float((ax > kw.get("k", 5.0) * base).mean())
