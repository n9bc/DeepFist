"""Tempo (WPM) estimation + normalization — inference-time, no retraining.

fldigi/MRP40 both continuously track WPM and make every timing decision relative
to it. Our neural model instead sees a fixed 6 s window at fixed resolution, so a
10 wpm and a 35 wpm signal look very different. If the model is weakest away from
its training-speed sweet spot, time-warping the audio toward that speed before
decoding can help — especially on odd-speed or drifting hand-sent fists.

estimate_wpm: from the CW-band keying envelope, dit ~ the base ON element; the
25th percentile of ON-run durations is a robust dit estimate (dits outnumber
dahs 1-unit vs 3-unit). normalize: resample so effective wpm -> target (pitch
shifts too, but the conditioner re-centers pitch to 600 Hz, so that's harmless).

Validate on real ARRL CER across 10-30 wpm (synthetic-only gains not trusted).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft, resample_poly
from fractions import Fraction

CW_LO, CW_HI = 550, 800
FRAME_MS = 5.0


def _tone_envelope(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr * FRAME_MS / 1000))
    nper = min(max(hop * 2, 512), x.size)
    f, _t, Z = stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None)
    mag = np.abs(Z)
    band = (f >= CW_LO) & (f <= CW_HI)
    sub = mag[band]
    bi = int(np.argmax(sub.mean(1)))
    env = sub[max(0, bi - 1):bi + 2].sum(0)
    return env / (env.max() + 1e-9), FRAME_MS / 1000.0


def estimate_wpm(x: np.ndarray, sr: int) -> float:
    """Estimate sending speed in WPM from ON-element durations. 0 if unclear."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    env, dt = _tone_envelope(x, sr)
    on = env > 0.4                                   # key-down frames
    if on.sum() < 3:
        return 0.0
    # run-lengths of consecutive ON frames
    d = np.diff(on.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(on)]
    runs = (ends - starts) * dt                      # seconds per ON element
    runs = runs[runs > 0]
    if runs.size < 3:
        return 0.0
    dit_s = np.percentile(runs, 25)                  # dit ~ shortest common element
    if dit_s <= 0:
        return 0.0
    return float(1200.0 / (dit_s * 1000.0))          # PARIS: wpm = 1200 / dit_ms


def normalize(x: np.ndarray, sr: int, target_wpm: float = 22.0,
              est_wpm: float | None = None,
              lo: float = 8.0, hi: float = 60.0) -> tuple[np.ndarray, float]:
    """Time-warp x so effective speed -> target_wpm. Returns (audio, est_wpm).

    No-op (returns x) if speed can't be estimated or is already near target.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    w = est_wpm if est_wpm is not None else estimate_wpm(x, sr)
    if w <= 0 or not (lo <= w <= hi):
        return x, w
    ratio = w / target_wpm                            # >1 => faster than target
    if abs(ratio - 1.0) < 0.06:
        return x, w
    frac = Fraction(ratio).limit_denominator(64)      # stretch time by `ratio`
    up, down = frac.numerator, frac.denominator
    if up == 0 or down == 0:
        return x, w
    return resample_poly(x, up, down).astype(np.float32), w
