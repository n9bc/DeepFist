"""Channel impairments: AWGN, QRN, QSB, flutter, frequency offset, QRM.

Impairment shapes are informed by WebMorseRunner (Unlicense) and reimplemented here.
"""
from dataclasses import dataclass
import numpy as np


def signal_power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x.astype(np.float64)))) + 1e-12


def estimate_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    return 10.0 * np.log10(signal_power(clean) / signal_power(noise))


def add_awgn(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = signal_power(audio)
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=len(audio))
    return (audio.astype(np.float64) + noise).astype(np.float32)


def apply_qsb(audio, sample_rate, rng, rate_hz=0.3, depth=0.6) -> np.ndarray:
    t = np.arange(len(audio)) / sample_rate
    phase = rng.uniform(0, 2 * np.pi)
    gain = 1.0 - depth * 0.5 * (1 - np.cos(2 * np.pi * rate_hz * t + phase))
    return (audio.astype(np.float64) * gain).astype(np.float32)


def apply_flutter(audio, sample_rate, rng, rate_hz=12.0, depth=0.3) -> np.ndarray:
    t = np.arange(len(audio)) / sample_rate
    phase = rng.uniform(0, 2 * np.pi)
    gain = 1.0 - depth * 0.5 * (1 - np.cos(2 * np.pi * rate_hz * t + phase))
    return (audio.astype(np.float64) * gain).astype(np.float32)


def apply_qrn(audio, sample_rate, rng, rate_per_s=3.0, amplitude=0.5) -> np.ndarray:
    out = audio.astype(np.float64).copy()
    n_crashes = rng.poisson(rate_per_s * len(audio) / sample_rate)
    for _ in range(int(n_crashes)):
        idx = int(rng.integers(0, len(out)))
        decay = int(0.01 * sample_rate)  # ~10 ms static crack
        env = np.exp(-np.arange(decay) / (decay / 4))
        burst = amplitude * rng.normal(0, 1, size=decay) * env
        end = min(idx + decay, len(out))
        out[idx:end] += burst[: end - idx]
    return out.astype(np.float32)


def apply_rx_filter(audio, sample_rate, center_hz, width_hz) -> np.ndarray:
    """Band-limit through a CW-width passband (raised-cosine skirts), FFT domain.

    Mimics a real receiver's CW filter: confines noise to the passband instead
    of the flat wideband AWGN, matching the spectrogram texture of on-air audio.
    """
    n = len(audio)
    if n < 4:
        return audio.astype(np.float32)
    X = np.fft.rfft(audio.astype(np.float64))
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    lo, hi = center_hz - width_hz / 2, center_hz + width_hz / 2
    skirt = max(20.0, width_hz * 0.25)
    mask = ((f >= lo) & (f <= hi)).astype(np.float64)
    ls = (f >= lo - skirt) & (f < lo)
    mask[ls] = 0.5 * (1 + np.cos(np.pi * (lo - f[ls]) / skirt))
    hs = (f > hi) & (f <= hi + skirt)
    mask[hs] = 0.5 * (1 + np.cos(np.pi * (f[hs] - hi) / skirt))
    return np.fft.irfft(X * mask, n=n).astype(np.float32)


def apply_freq_offset(audio, sample_rate, offset_hz) -> np.ndarray:
    # Real-signal frequency shift via analytic signal.
    from numpy.fft import fft, ifft
    n = len(audio)
    analytic = ifft(fft(audio) * _hilbert_mask(n))
    t = np.arange(n) / sample_rate
    shifted = np.real(analytic * np.exp(2j * np.pi * offset_hz * t))
    return shifted.astype(np.float32)


def _hilbert_mask(n: int) -> np.ndarray:
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return h


@dataclass
class ChannelConfig:
    qsb: bool = True
    flutter: bool = False
    qrn: bool = True
    freq_offset: bool = True
    rx_filter: bool = True
    qsb_prob: float = 0.65
    flutter_prob: float = 0.2
    qrn_prob: float = 0.5
    offset_prob: float = 0.5
    rx_filter_prob: float = 0.6


def degrade(audio, sample_rate, snr_db, rng, config: ChannelConfig,
            pitch_hz: float | None = None) -> np.ndarray:
    x = audio.astype(np.float32)
    if config.qsb and rng.random() < config.qsb_prob:
        # Real off-air QSB is often slow AND deep — measured ~0.11 Hz (9 s period)
        # fading to ~5 % of peak on real recordings. Cover that: rate down to
        # 0.05 Hz and depth up to 0.97 (trough ~3 %), not just shallow/fast fades.
        x = apply_qsb(x, sample_rate, rng,
                      rate_hz=float(rng.uniform(0.05, 1.0)),
                      depth=float(rng.uniform(0.3, 0.97)))
    if config.flutter and rng.random() < config.flutter_prob:
        x = apply_flutter(x, sample_rate, rng, rate_hz=float(rng.uniform(5, 20)))
    if config.freq_offset and rng.random() < config.offset_prob:
        x = apply_freq_offset(x, sample_rate, float(rng.uniform(-80, 80)))
    if config.qrn and rng.random() < config.qrn_prob:
        x = apply_qrn(x, sample_rate, rng)
    x = add_awgn(x, snr_db, rng)
    # Receiver CW filter last: shapes signal AND noise into a real passband.
    if config.rx_filter and pitch_hz is not None and rng.random() < config.rx_filter_prob:
        center = pitch_hz + float(rng.uniform(-40, 40))
        width = float(rng.uniform(200, 550))
        x = apply_rx_filter(x, sample_rate, center, width)
    return np.clip(x, -1.0, 1.0).astype(np.float32)
