import numpy as np
from deepfist.synth.tone import envelope_to_audio


def test_pure_tone_peaks_at_expected_frequency():
    sr = 8000
    env = np.ones(sr, dtype=np.float32)  # 1 s key-down
    audio = envelope_to_audio(env, sr, 650.0)
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    assert abs(freqs[np.argmax(spectrum)] - 650.0) < 5.0


def test_amplitude_bounded_and_float32():
    sr = 8000
    env = np.ones(sr, dtype=np.float32)
    audio = envelope_to_audio(env, sr, 650.0)
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0001


def test_silence_envelope_gives_silence():
    audio = envelope_to_audio(np.zeros(800, dtype=np.float32), 8000, 650.0)
    assert np.allclose(audio, 0.0)
