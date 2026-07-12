"""Audio -> log-magnitude spectrogram (band-limited, globally standardized)."""
import numpy as np
import torch

# DeepCW-style hi-res front-end: 3200 Hz gives 12.5 Hz FFT bins, so the CW tone
# is sharp and nearby QRM separates (vs 31 Hz bins at 8 kHz).
N_FFT = 256
HOP = 48
BAND_LO_HZ = 400
BAND_HI_HZ = 1200


SAMPLE_RATE = 3200        # canonical model input rate


def _band_bins(sample_rate: int) -> tuple[int, int]:
    hz_per_bin = sample_rate / N_FFT
    lo = int(np.ceil(BAND_LO_HZ / hz_per_bin))
    hi = int(np.floor(BAND_HI_HZ / hz_per_bin)) + 1  # exclusive upper
    return lo, hi


FREQ_BINS = _band_bins(SAMPLE_RATE)[1] - _band_bins(SAMPLE_RATE)[0]   # 65


def _as_tensor(audio) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio.astype(np.float32))
    return audio.to(torch.float32)


def audio_to_spectrogram(audio, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    x = _as_tensor(audio)
    window = torch.hann_window(N_FFT, device=x.device)
    stft = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=window,
                      center=True, return_complex=True)
    mag = stft.abs()                       # [freq, T]
    lo, hi = _band_bins(sample_rate)
    mag = mag[lo:hi]                        # [65, T] at 3200 Hz
    spec = torch.log1p(mag)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)   # global standardize
    return spec.to(torch.float32)


def audio_batch_to_spectrogram(batch, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    specs = [audio_to_spectrogram(batch[i], sample_rate) for i in range(len(batch))]
    return torch.stack(specs).unsqueeze(1)   # [B,1,F,T]
