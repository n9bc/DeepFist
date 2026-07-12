import numpy as np
import torch
from deepfist.features.spectrogram import audio_to_spectrogram, audio_batch_to_spectrogram


def _tone(sr=3200, secs=6.0, f=650.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.8 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_shape_and_dtype():
    spec = audio_to_spectrogram(_tone())
    assert spec.dtype == torch.float32
    assert spec.shape[0] == 65   # 400-1200 Hz at 12.5 Hz bins = bins 32..96
    assert 390 <= spec.shape[1] <= 410   # 6s @ 3200 Hz, hop 48


def test_tone_energy_in_expected_bin():
    spec = audio_to_spectrogram(_tone(f=650.0))
    # 650 Hz -> bin 650/12.5=52 absolute; band starts at bin 32 -> local index ~20.
    profile = spec.mean(dim=1)
    peak = int(profile.argmax())
    assert 18 <= peak <= 22


def test_global_standardized():
    spec = audio_to_spectrogram(_tone())
    assert abs(float(spec.mean())) < 1e-4
    assert abs(float(spec.std()) - 1.0) < 1e-2


def test_silence_is_finite():
    spec = audio_to_spectrogram(np.zeros(19200, dtype=np.float32))
    assert torch.isfinite(spec).all()


def test_batch_shape():
    batch = np.stack([_tone(), _tone(f=700.0)])
    out = audio_batch_to_spectrogram(batch)
    assert out.shape[0] == 2 and out.shape[1] == 1 and out.shape[2] == 65
