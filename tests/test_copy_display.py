"""Unit tests for the copy-display pipeline pieces: squelch (keying gate),
despike (impulse blanker), tempo (WPM estimator), cw_lm (LM stub protect-list),
and the Farnsworth-geometry augmentation. Pure synthesis — no model, no radio."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import squelch as SQ
import despike as DS
import tempo as TP
import cw_lm as LM
from deepfist.morse.timing import wpm_to_timing
from deepfist.synth.keyer import text_to_segments, segments_to_envelope
from deepfist.synth.tone import envelope_to_audio
from deepfist.synth.generator import generate, GenConfig

SR = 3200


def keyed_cw(text="CQ CQ DE W1AW W1AW K", wpm=20, pitch=700.0, seconds=6.0):
    """Real CW audio at a known speed (uses the project's own keyer)."""
    segs = text_to_segments(text, wpm_to_timing(wpm))
    env = segments_to_envelope(segs, SR, rise=0.005)
    a = envelope_to_audio(env, SR, pitch)
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    out[:min(n, len(a))] = a[:n]
    return out


def steady_tone(pitch=700.0, seconds=6.0, agc_breathe=0.3):
    """Dead-air stand-in: unmodulated tone with slow AGC-like breathing."""
    t = np.arange(int(seconds * SR)) / SR
    amp = 1.0 + agc_breathe * np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(0)
    return (amp * np.sin(2 * np.pi * pitch * t)
            + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


# ---------------- squelch ----------------

def test_squelch_rejects_steady_tone():
    ok, score = SQ.has_signal(steady_tone(), SR)
    assert not ok and score < SQ.DEFAULT_THRESH


def test_squelch_passes_keyed_cw():
    ok, score = SQ.has_signal(keyed_cw(), SR)
    assert ok and score > SQ.DEFAULT_THRESH


def test_squelch_calibrate_floor():
    thresh, floor = SQ.calibrate(steady_tone(seconds=18.0), SR)
    assert thresh >= SQ.DEFAULT_THRESH
    assert floor < thresh


# ---------------- despike ----------------

def test_despike_neutral_on_clean_cw():
    a = keyed_cw()
    y = DS.despike(a, SR)
    assert y.shape == a.shape
    assert np.abs(y - a).max() < 0.15 * np.abs(a).max()   # essentially unchanged


def test_despike_blanks_impulse():
    a = keyed_cw()
    peak = np.abs(a).max()
    b = a.copy()
    b[SR] += 12 * peak                                     # 1-sample crash
    y = DS.despike(b, SR)
    assert np.abs(y[SR]) < 2 * peak                        # spike removed
    # signal elsewhere intact
    assert np.abs(y[3 * SR:] - a[3 * SR:]).max() < 0.15 * peak


# ---------------- tempo ----------------

@pytest.mark.parametrize("wpm", [15, 20, 30])
def test_estimate_wpm_close(wpm):
    a = keyed_cw("PARIS PARIS PARIS PARIS", wpm=wpm, seconds=8.0)
    est = TP.estimate_wpm(a, SR)
    assert est > 0
    assert abs(est - wpm) / wpm < 0.35                     # within 35% (robust estimator)


# ---------------- cw_lm ----------------

def test_lm_protects_rare_tokens():
    lm = LM.load_default()
    assert lm.is_protected("KE9XYZ")        # callsign
    assert lm.is_protected("599")           # number
    assert lm.is_protected("EN52")          # grid
    assert not lm.is_protected("QTH")
    assert not lm.is_protected("WEATHER")


def test_lm_scores_finite_and_trainable():
    lm = LM.load_default()
    p0 = lm.logprob("RIG", "MY")
    lm.train(["MY RIG IS A K3", "MY RIG RUNS QRP"])
    p1 = lm.logprob("RIG", "MY")
    assert np.isfinite(p0) and np.isfinite(p1)
    assert p1 > p0                                          # training raised P(RIG|MY)


# ---------------- Farnsworth augmentation ----------------

def test_farnsworth_timing_stretches_only_gaps():
    plain = wpm_to_timing(20)
    farns = wpm_to_timing(20, farnsworth_wpm=10)
    assert farns.dot == plain.dot and farns.dash == plain.dash
    assert farns.element_gap == plain.element_gap
    assert farns.char_gap > plain.char_gap                  # ratios DISTORTED
    assert farns.word_gap > plain.word_gap


def test_generator_farnsworth_prob_off_is_deterministic_default():
    a = generate(seed=7)
    b = generate(seed=7, config=GenConfig())
    assert a.label == b.label


def test_generator_farnsworth_produces_valid_clips():
    cfg = GenConfig(farnsworth_prob=1.0, impair=False, qrm=False)
    for s in range(3):
        x = generate(seed=s, config=cfg)
        assert len(x.audio) == int(cfg.window_s * cfg.sample_rate)
        assert len(x.label) > 0


# ---------------- per-gap hesitation ----------------

def test_hesitation_stretches_only_char_word_gaps():
    from deepfist.synth.fist import apply_fist, FistParams
    rng = np.random.default_rng(0)
    segs = text_to_segments("CQ CQ", wpm_to_timing(20))
    base = apply_fist(segs, np.random.default_rng(1), FistParams(jitter_sigma=0.0))
    hes = apply_fist(segs, np.random.default_rng(1),
                     FistParams(jitter_sigma=0.0, hesitation=2.0))
    dit = min(s.duration for s in segs if s.on)
    for b, h, orig in zip(base, hes, segs):
        if orig.on or orig.duration <= 2.2 * dit:
            assert abs(b.duration - h.duration) < 1e-9    # elements + element-gaps untouched
        else:
            assert h.duration >= b.duration               # char/word gaps stretched


def test_generator_hesitation_produces_valid_clips():
    cfg = GenConfig(hesitation_max=2.0, impair=False, qrm=False)
    for s in range(3):
        x = generate(seed=s, config=cfg)
        assert len(x.audio) == int(cfg.window_s * cfg.sample_rate)
        assert len(x.label) > 0
