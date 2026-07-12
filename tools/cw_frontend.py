"""fldigi-inspired CW front-end conditioning for DeepFist (reference impl to port
to Rust). Isolates ONE CW signal from a busy passband and normalizes it:

  1. AGC  -- normalize level (fixes out-of-scale live audio, e.g. peak ~20).
  2. Tone detect (AFC) -- find the dominant CW tone in 400-1200 Hz.
  3. Matched narrow bandpass -- complex downconvert at that tone + low-pass
     (bandwidth ~ CW width), rejecting adjacent QRM.
  4. Re-center -- put the isolated tone at a fixed canonical pitch so the model
     always sees a consistent pitch.

This is a single-signal isolator; the neural model then classifies the clean tone.
"""
import numpy as np
from scipy.signal import welch


def detect_tone(audio: np.ndarray, sr: int, lo=400.0, hi=1200.0) -> float:
    f, p = welch(audio, sr, nperseg=min(8192, len(audio)))
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return 600.0
    return float(f[band][p[band].argmax()])


def condition(audio: np.ndarray, sr: int, out_pitch=600.0, bw_hz=90.0,
              tone_hz: float | None = None) -> np.ndarray:
    """Isolate + normalize + re-center one CW signal. Returns audio at `sr`."""
    x = audio.astype(np.float32)
    # 1. AGC (unit RMS) — removes the wild live-audio scale before anything else.
    rms = np.sqrt((x * x).mean()) + 1e-9
    x = x / rms
    # 2. tone detect (AFC)
    tone = tone_hz if tone_hz is not None else detect_tone(x, sr)
    # 3. complex downconvert to baseband + one-pole low-pass (matched-ish)
    n = len(x)
    t = np.arange(n) / sr
    bb = x * np.exp(-2j * np.pi * tone * t)
    alpha = 1.0 - np.exp(-2 * np.pi * (bw_hz / 2) / sr)  # 1-pole LPF coeff
    # two cascaded 1-pole LPFs for steeper skirts
    y = np.empty(n, dtype=np.complex64)
    s = 0.0 + 0.0j
    for i in range(n):
        s += alpha * (bb[i] - s)
        y[i] = s
    s2 = 0.0 + 0.0j
    for i in range(n):
        s2 += alpha * (y[i] - s2)
        y[i] = s2
    # 4. re-center at a fixed canonical pitch, take real part
    out = (y * np.exp(2j * np.pi * out_pitch * t)).real.astype(np.float32)
    peak = np.abs(out).max() + 1e-9
    return (out / peak).astype(np.float32)
