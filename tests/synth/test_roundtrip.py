import numpy as np
from deepfist.synth.generator import generate, GenConfig
from deepfist.synth.channel import ChannelConfig
from deepfist.synth.envelope_decode import decode_clean_envelope


def test_clean_signal_round_trips():
    # No impairments, fixed pitch/wpm via a high-SNR clean config.
    cfg = GenConfig(impair=False)
    hits = 0
    trials = 20
    for seed in range(trials):
        s = generate(seed=seed, config=cfg)
        decoded = decode_clean_envelope(
            s.audio, cfg.sample_rate, s.meta["pitch_hz"], s.meta["wpm"])
        # Allow small edge errors; require high character overlap.
        if decoded.replace(" ", "") == s.label.replace(" ", ""):
            hits += 1
    assert hits >= int(trials * 0.7)  # generator is self-consistent on clean audio
