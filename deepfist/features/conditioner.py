"""Vectorized CW front-end conditioner — the training-side twin of the live
Rust conditioner in diddle (`cw_neural.rs::Conditioner::condition`).

Live inference decimates radio audio to 3200 Hz, then runs this conditioner
(AGC -> tone AFC -> matched narrow bandpass -> re-center to 600 Hz -> peak-norm)
*before* the spectrogram. Training historically fed RAW audio, so the model saw
a different distribution live than in training. Applying the SAME conditioning to
training data closes that gap and matches deployment levels.

This is a numpy/scipy vectorization (scipy.lfilter for the 1-pole cascade) of the
per-sample Rust loop, ~1000x faster so it runs on-the-fly in the DataLoader.

Constants mirror cw_neural.rs: SR=3200, TONE_NFFT=4096, OUT_PITCH=600, BW=90 Hz.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.signal import lfilter

SR = 3200
TONE_NFFT = 4096
OUT_PITCH = 600.0
COND_BW_HZ = 90.0
BAND_LO_HZ = 400.0
BAND_HI_HZ = 1200.0


def detect_tone(audio: np.ndarray, sr: int = SR) -> float:
    """Dominant CW tone (Hz) in 400-1200 Hz via a single FFT (matches Rust)."""
    n = min(len(audio), TONE_NFFT)
    if n < 8:
        return OUT_PITCH
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(audio[:n] * w))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)
    if not band.any():
        return OUT_PITCH
    idx = np.where(band)[0]
    return float(freqs[idx[spec[idx].argmax()]])


def condition(audio: np.ndarray, sr: int = SR, tone_hz: float | None = None) -> np.ndarray:
    """Isolate + normalize one CW signal; returns real audio at `sr`, peak≈1.

    Vectorized twin of cw_neural.rs::condition. Expects audio already at `sr`
    (== model rate, 3200 Hz) for a faithful match to live inference."""
    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    if n < TONE_NFFT:
        return x
    # 1. AGC — unit RMS
    rms = np.sqrt((x * x).mean()) + 1e-9
    x = x / rms
    # 2. tone AFC
    tone = tone_hz if tone_hz is not None else detect_tone(x, sr)
    # 3. complex downconvert + two cascaded 1-pole LPFs
    k = np.arange(n, dtype=np.float64)
    bb = x * np.exp(-2j * np.pi * tone * k / sr)
    alpha = 1.0 - np.exp(-2.0 * np.pi * (COND_BW_HZ * 0.5) / sr)
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    y = lfilter(b, a, bb)
    y = lfilter(b, a, y)
    # 4. re-center at OUT_PITCH, take real part, peak-normalize
    out = np.real(y * np.exp(2j * np.pi * OUT_PITCH * k / sr)).astype(np.float32)
    peak = np.abs(out).max() + 1e-9
    return (out / peak).astype(np.float32)


def maybe_condition(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Apply conditioning iff DEEPFIST_CONDITION is truthy. Shared gate so the
    training loaders and eval tools stay in lockstep via one env var."""
    if os.environ.get("DEEPFIST_CONDITION", "").lower() in ("1", "true", "yes", "on"):
        return condition(audio, sr)
    return audio
