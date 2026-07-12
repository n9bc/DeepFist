"""MP3 encode/decode round-trip augmentation (lossy-codec artifacts).

Real training/eval audio (ARRL W1AW practice, K5ZD archive, most on-air captures)
reaches us as **MP3**, so the received CW carries lossy-codec artifacts: pre-echo
smearing around keying edges, quantisation noise shaped by the psychoacoustic
model, and band-limiting from the encoder's bit allocation. Our synthetic audio was
never encoded, so it lacks that texture — a suspected slice of the synthetic->real
gap (HANDOFF §17.6).

`mp3_roundtrip` pipes audio through ffmpeg (libmp3lame) at a chosen bitrate and back.
Our model rate (3200 Hz) is below MP3's minimum sample rate, so we resample up to a
codec-legal rate for the encode, then back to `sr` — mirroring the real chain
(48 kHz capture -> MP3 -> decimate). Lower bitrates = heavier artifacts.

ffmpeg must be on PATH (it is on the dev box; CI/tests skip when absent).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# Sample rate used for the actual MP3 encode. 3200 Hz is below MP3's floor, so we
# up-sample to a legal MPEG rate first. 8 kHz (MPEG-2.5) keeps the Nyquist tight so
# the encoder's bit budget concentrates near our 400-1200 Hz band -> audible artifacts.
_MP3_SR = 8000


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mp3_roundtrip(audio: np.ndarray, sr: int, bitrate_kbps: int = 32) -> np.ndarray:
    """Encode `audio` to MP3 at `bitrate_kbps` and decode back to `sr`.

    Returns float32 audio the same length as the input (trimmed/padded to match).
    If ffmpeg is missing the input is returned unchanged so callers never crash.
    """
    if not ffmpeg_available():
        return audio.astype(np.float32)
    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    if n < 8:
        return x
    peak = float(np.abs(x).max()) + 1e-9
    x_norm = x / peak                       # avoid int16 clipping; restored after
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        in_wav, mp3, out_wav = d / "in.wav", d / "a.mp3", d / "out.wav"
        wavfile.write(in_wav, sr, (x_norm * 32767.0).astype(np.int16))
        common = ["-hide_banner", "-loglevel", "error", "-y"]
        # encode (ffmpeg resamples sr -> _MP3_SR internally)
        subprocess.run(["ffmpeg", *common, "-i", str(in_wav), "-ar", str(_MP3_SR),
                        "-b:a", f"{bitrate_kbps}k", str(mp3)], check=True)
        # decode back to the model rate
        subprocess.run(["ffmpeg", *common, "-i", str(mp3), "-ar", str(sr),
                        str(out_wav)], check=True)
        _, y = wavfile.read(out_wav)
    y = (y.astype(np.float32) / 32768.0) * peak      # restore original scale
    if len(y) >= n:                                  # align length to the input
        y = y[:n]
    else:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def maybe_mp3(audio: np.ndarray, sr: int, rng: np.random.Generator,
              prob: float, bitrates_kbps) -> np.ndarray:
    """Apply an MP3 round-trip with probability `prob` at a random bitrate.

    Returns the input array **unchanged** (same object) when skipped, so callers
    can detect application via identity.
    """
    if prob <= 0.0 or not bitrates_kbps or rng.random() >= prob:
        return audio
    br = int(rng.choice(np.asarray(bitrates_kbps)))
    return mp3_roundtrip(audio, sr, br)
