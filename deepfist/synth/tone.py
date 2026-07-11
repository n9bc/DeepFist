"""Modulate an on/off envelope onto a (optionally drifting) sine carrier."""
import numpy as np


def envelope_to_audio(envelope: np.ndarray, sample_rate: int, pitch_hz: float,
                      drift_hz: float = 0.0) -> np.ndarray:
    n = len(envelope)
    t = np.arange(n) / sample_rate
    if drift_hz:
        # Slow sinusoidal frequency wander over the clip; integrate to phase.
        drift = drift_hz * np.sin(2 * np.pi * 0.3 * t)
        phase = 2 * np.pi * np.cumsum(pitch_hz + drift) / sample_rate
    else:
        phase = 2 * np.pi * pitch_hz * t
    audio = envelope.astype(np.float64) * np.sin(phase)
    return audio.astype(np.float32)
