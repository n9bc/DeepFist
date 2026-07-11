import numpy as np
from deepfist.synth.text import random_message
from deepfist.morse.alphabet import text_to_tokens


def test_message_tokenizes_and_respects_length():
    rng = np.random.default_rng(0)
    for _ in range(200):
        msg = random_message(rng, max_tokens=30)
        toks = text_to_tokens(msg)      # must not raise
        assert 0 < len(toks) <= 30


def test_message_is_seed_deterministic():
    a = random_message(np.random.default_rng(42))
    b = random_message(np.random.default_rng(42))
    assert a == b
