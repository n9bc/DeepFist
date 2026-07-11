import numpy as np
from deepfist.synth.keyer import Segment
from deepfist.synth.fist import apply_fist, FistParams


def _segs():
    return [Segment(True, 0.06), Segment(False, 0.06), Segment(True, 0.18)]


def test_fist_is_deterministic_per_seed():
    a = apply_fist(_segs(), np.random.default_rng(1), FistParams())
    b = apply_fist(_segs(), np.random.default_rng(1), FistParams())
    assert [s.duration for s in a] == [s.duration for s in b]


def test_fist_keeps_durations_positive_and_count():
    out = apply_fist(_segs(), np.random.default_rng(3), FistParams(jitter_sigma=0.2))
    assert len(out) == 3
    assert all(s.duration > 0 for s in out)


def test_zero_jitter_zero_weight_is_identity():
    src = _segs()
    out = apply_fist(src, np.random.default_rng(0), FistParams(jitter_sigma=0.0, weight=0.0))
    assert [s.duration for s in out] == [s.duration for s in src]


def test_positive_weight_lengthens_on_segments():
    src = _segs()
    out = apply_fist(src, np.random.default_rng(0), FistParams(jitter_sigma=0.0, weight=0.2))
    on_src = sum(s.duration for s in src if s.on)
    on_out = sum(s.duration for s in out if s.on)
    assert on_out > on_src
