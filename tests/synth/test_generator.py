import numpy as np
from deepfist.synth.generator import generate, GenConfig
from deepfist.morse.alphabet import text_to_tokens, TOKENS


def test_sample_shape_and_label():
    cfg = GenConfig()
    s = generate(seed=0, config=cfg)
    assert s.audio.dtype == np.float32
    assert len(s.audio) == int(cfg.window_s * cfg.sample_rate)
    assert np.max(np.abs(s.audio)) <= 1.0001
    assert len(s.label) > 0


def test_generation_is_seed_deterministic():
    a = generate(seed=123)
    b = generate(seed=123)
    assert np.array_equal(a.audio, b.audio)
    assert a.label == b.label


def test_label_tokens_are_all_valid():
    for seed in range(50):
        s = generate(seed=seed)
        for tok in text_to_tokens(s.label):
            assert tok in TOKENS


def test_message_fits_window():
    # keyed message duration must not exceed the window (leaves room for CTC).
    for seed in range(50):
        s = generate(seed=seed)
        assert s.meta["keyed_duration_s"] <= s.meta["window_s"]


def test_realism_knobs_default_is_a_noop():
    # New augmentation knobs default to current behaviour -> byte-identical output.
    a = generate(seed=7)
    b = generate(seed=7, config=GenConfig())
    assert np.array_equal(a.audio, b.audio)
    assert a.meta["mp3"] is False


def test_morphology_knobs_change_audio_but_stay_valid():
    cfg = GenConfig(dahdit_jitter=0.4, gap_scale_range=(0.65, 1.15),
                    rise_range=(0.002, 0.015))
    s = generate(seed=7, config=cfg)
    base = generate(seed=7)
    assert not np.array_equal(s.audio, base.audio)      # knobs had an effect
    assert len(s.audio) == int(cfg.window_s * cfg.sample_rate)
    for tok in text_to_tokens(s.label):
        assert tok in TOKENS
