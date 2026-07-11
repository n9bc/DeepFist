import math
from deepfist.morse.timing import wpm_to_timing


def test_paris_dot_at_20wpm():
    t = wpm_to_timing(20)
    assert math.isclose(t.dot, 0.06, rel_tol=1e-9)      # 1.2/20
    assert math.isclose(t.dash, 0.18, rel_tol=1e-9)     # 3*dot
    assert math.isclose(t.char_gap, 0.18, rel_tol=1e-9) # 3*dot
    assert math.isclose(t.word_gap, 0.42, rel_tol=1e-9) # 7*dot


def test_dot_scales_inversely_with_wpm():
    assert math.isclose(wpm_to_timing(10).dot, 2 * wpm_to_timing(20).dot, rel_tol=1e-9)


def test_farnsworth_stretches_gaps_only():
    fast = wpm_to_timing(30)
    fw = wpm_to_timing(30, farnsworth_wpm=15)
    assert math.isclose(fw.dot, fast.dot, rel_tol=1e-9)       # element speed unchanged
    assert fw.char_gap > fast.char_gap                        # gaps stretched
    assert fw.word_gap > fast.word_gap
