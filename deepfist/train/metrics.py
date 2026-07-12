"""CER metric and per-SNR benchmark evaluation."""
import torch

from deepfist.synth.generator import generate, GenConfig
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.features.conditioner import maybe_condition
from deepfist.model.decode import greedy_ctc_decode


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def cer(pred: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(pred, target) / len(target)


def evaluate_per_snr(model, snr_points, clips_per_point, gen_config=None,
                     device="cpu") -> dict:
    base = gen_config or GenConfig()
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snr_points:
            cfg = GenConfig(sample_rate=base.sample_rate, window_s=base.window_s,
                            wpm_range=base.wpm_range, pitch_range=base.pitch_range,
                            snr_range=(snr, snr), impair=True, channel=base.channel)
            total = 0.0
            for i in range(clips_per_point):
                s = generate(seed=10_000 + int(snr) * 100 + i, config=cfg)
                spec = audio_to_spectrogram(maybe_condition(s.audio, cfg.sample_rate), cfg.sample_rate)
                lp = model(spec.unsqueeze(0).unsqueeze(0).to(device))
                pred = greedy_ctc_decode(lp)[0]
                total += cer(pred, s.label)
            results[float(snr)] = total / clips_per_point
    return results
