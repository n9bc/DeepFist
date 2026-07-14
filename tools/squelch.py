"""Keying-based squelch (signal-presence gate) for the live CW decoder.

Why not a level gate: in CW mode Lyra's narrow filter + AGC produce a steady
~700 Hz "artifact" tone whose broadband level is indistinguishable from a real
signal (dead air rms ~3.5, signal rms ~3.7-6.6). A level/peak gate cannot tell
them apart, so the decoder hallucinates stray characters on empty frequencies.

What DOES separate them is KEYING. A real CW signal is on/off: its tone bin
swings from a loud key-down state to the true noise floor during gaps. The
artifact is a STEADY tone (AGC only breathes it ~4x). So per CW-band bin we
measure keying_ratio = p90/p10 of its amplitude envelope over time, and take the
max across bins (the most-keyed tone). Ground truth (30s clips, per 6s window):

    dead air / untuned : keying_ratio 3.4 - 3.9   -> NO SIGNAL
    tuned-in CW signal : keying_ratio 41  - 73    -> SIGNAL

A threshold of ~12 sits in the wide empty gap. The floor (~4) is stable, but it
can be re-calibrated per band with calibrate() (sample dead air, threshold =
floor * margin) per the operator's per-band-change workflow.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

FRAME_MS = 10.0            # envelope frame period
CW_LO, CW_HI = 550, 800    # CW audio-tone search band (Hz) — Lyra CW pitch ~700 Hz
DEFAULT_THRESH = 12.0      # keying_ratio below this = no signal


def keying_ratio(audio: np.ndarray, sr: int) -> float:
    """Max over CW-band bins of p90/p10 of the amplitude envelope.

    Steady tone / noise -> ~4. Keyed CW signal -> tens. Sample-rate agnostic.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    if a.size < 256:
        return 0.0
    hop = max(1, int(sr * FRAME_MS / 1000))
    nper = min(max(hop * 2, 512), a.size)
    f, _t, Z = stft(a, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None)
    mag = np.abs(Z)
    band = (f >= CW_LO) & (f <= CW_HI)
    if not band.any() or mag.shape[1] < 4:
        return 0.0
    sub = mag[band]                                   # [bins, frames]
    # smooth over 3 adjacent bins (a CW tone spans a couple bins) for robustness
    if sub.shape[0] >= 3:
        sub = np.stack([sub[max(0, i - 1):i + 2].sum(0) for i in range(sub.shape[0])])
    p10 = np.percentile(sub, 10, axis=1)
    p90 = np.percentile(sub, 90, axis=1)
    return float(np.max(p90 / (p10 + 1e-3)))


def has_signal(audio: np.ndarray, sr: int, thresh: float = DEFAULT_THRESH) -> tuple[bool, float]:
    """(signal_present, keying_ratio_score) for one window."""
    score = keying_ratio(audio, sr)
    return score >= thresh, score


def calibrate(dead_air: np.ndarray, sr: int, margin: float = 3.0,
              win_s: float = 6.0, hop_s: float = 3.0) -> tuple[float, float]:
    """From a dead-air reference, return (suggested_thresh, observed_floor).

    floor = max keying_ratio seen across windows of the no-signal sample;
    suggested threshold = floor * margin (clamped to >= DEFAULT_THRESH).
    """
    a = np.asarray(dead_air, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    win, hop = int(win_s * sr), int(hop_s * sr)
    scores = [keying_ratio(a[s:s + win], sr)
              for s in range(0, max(1, len(a) - win + 1), hop)]
    floor = max(scores) if scores else 0.0
    return max(DEFAULT_THRESH, floor * margin), floor
