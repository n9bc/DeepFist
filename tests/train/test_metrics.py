from deepfist.train.metrics import cer


def test_exact_match_zero():
    assert cer("CQ TEST", "CQ TEST") == 0.0


def test_single_substitution():
    assert abs(cer("CAT", "COT") - 1/3) < 1e-9


def test_insertion_and_deletion():
    assert abs(cer("CQ", "CQX") - 1/3) < 1e-9   # one deletion vs target len 3
    assert abs(cer("ABC", "AC") - 1/2) < 1e-9   # one insertion vs target len 2


def test_empty_target():
    assert cer("", "") == 0.0
    assert cer("X", "") == 1.0
