import numpy as np
import pytest

from deepfist.synth.codec import ffmpeg_available, maybe_mp3, mp3_roundtrip


def _tone(sr=3200, secs=1.0, hz=600.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_maybe_mp3_skips_when_prob_zero_returns_same_object():
    x = _tone()
    rng = np.random.default_rng(0)
    assert maybe_mp3(x, 3200, rng, prob=0.0, bitrates_kbps=(32,)) is x


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_mp3_roundtrip_preserves_length_and_tone():
    x = _tone()
    y = mp3_roundtrip(x, 3200, bitrate_kbps=32)
    assert y.dtype == np.float32
    assert len(y) == len(x)
    # A 600 Hz tone survives MP3: dominant FFT bin stays near 600 Hz.
    freqs = np.fft.rfftfreq(len(y), 1.0 / 3200)
    peak = freqs[np.abs(np.fft.rfft(y)).argmax()]
    assert abs(peak - 600.0) < 25.0
