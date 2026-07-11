import numpy as np
from deepfist.synth import channel as ch


def _tone(sr=8000, secs=2.0, f=650.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.8 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_awgn_hits_target_snr_within_1db():
    rng = np.random.default_rng(0)
    clean = _tone()
    for target in (10.0, 0.0, -6.0):
        noisy = ch.add_awgn(clean, target, rng)
        measured = ch.estimate_snr_db(clean, noisy - clean)
        assert abs(measured - target) < 1.0


def test_freq_offset_shifts_peak():
    sr = 8000
    clean = _tone(sr)
    shifted = ch.apply_freq_offset(clean, sr, 100.0)
    f = np.fft.rfftfreq(len(shifted), 1 / sr)
    peak = f[np.argmax(np.abs(np.fft.rfft(shifted)))]
    assert abs(peak - 750.0) < 6.0  # 650 + 100


def test_degrade_is_seed_deterministic_and_bounded():
    cfg = ch.ChannelConfig()
    a = ch.degrade(_tone(), 8000, 3.0, np.random.default_rng(7), cfg)
    b = ch.degrade(_tone(), 8000, 3.0, np.random.default_rng(7), cfg)
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) <= 1.0001


def test_qsb_modulates_amplitude():
    sr = 8000
    clean = _tone(sr, secs=4.0)
    faded = ch.apply_qsb(clean, sr, np.random.default_rng(0), rate_hz=0.5, depth=0.8)
    env = np.abs(faded)
    # Amplitude varies substantially across the clip under deep QSB.
    assert env[: sr].mean() / (env[-sr:].mean() + 1e-9) > 1.2 or \
           env[-sr:].mean() / (env[: sr].mean() + 1e-9) > 1.2
