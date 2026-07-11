import numpy as np
from deepfist.synth.keyer import text_to_segments, segments_to_envelope
from deepfist.morse.timing import wpm_to_timing


def test_single_dot_has_one_on_segment():
    segs = text_to_segments("E", wpm_to_timing(20))  # E == "."
    on = [s for s in segs if s.on]
    assert len(on) == 1
    assert np.isclose(on[0].duration, 0.06)  # dot at 20 wpm


def test_prosign_runs_together_no_char_gap():
    t = wpm_to_timing(20)
    # <SK> is one token of 6 elements => 5 internal element gaps, zero char gaps inside.
    segs = text_to_segments("<SK>", t)
    on = [s for s in segs if s.on]
    assert len(on) == 6
    internal_gaps = [s for s in segs if not s.on][:-1]  # drop any trailing
    assert all(np.isclose(g.duration, t.element_gap) for g in internal_gaps)


def test_envelope_is_click_free_and_bounded():
    t = wpm_to_timing(20)
    env = segments_to_envelope(text_to_segments("PARIS", t), 8000)
    assert env.dtype == np.float32
    assert env.min() >= 0.0 and env.max() <= 1.0001
    # No hard 0->1 jumps: max sample-to-sample delta is small due to ramping.
    assert np.max(np.abs(np.diff(env))) < 0.2


def test_envelope_length_matches_total_duration():
    t = wpm_to_timing(20)
    segs = text_to_segments("E", t)
    total = sum(s.duration for s in segs)
    env = segments_to_envelope(segs, 8000)
    assert abs(len(env) - round(total * 8000)) <= 1
