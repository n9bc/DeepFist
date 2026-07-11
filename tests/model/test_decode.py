import torch
from deepfist.model.decode import greedy_ctc_decode, ids_to_text, BLANK_ID
from deepfist.morse.alphabet import TOKEN_TO_ID, TOKENS


def _logits_from_path(path_ids, n_classes=len(TOKENS)):
    # Build [T,1,C] where each frame's argmax is path_ids[t].
    T = len(path_ids)
    lp = torch.full((T, 1, n_classes), -10.0)
    for t, cid in enumerate(path_ids):
        lp[t, 0, cid] = 0.0
    return lp


def test_collapse_and_drop_blank():
    H, I = TOKEN_TO_ID["H"], TOKEN_TO_ID["I"]
    path = [H, H, BLANK_ID, I, I]
    assert greedy_ctc_decode(_logits_from_path(path)) == ["HI"]


def test_blank_only_is_empty():
    assert greedy_ctc_decode(_logits_from_path([BLANK_ID, BLANK_ID])) == [""]


def test_space_token_survives():
    A, SP = TOKEN_TO_ID["A"], TOKEN_TO_ID[" "]
    path = [A, BLANK_ID, SP, BLANK_ID, A]
    assert greedy_ctc_decode(_logits_from_path(path)) == ["A A"]


def test_ids_to_text():
    assert ids_to_text([TOKEN_TO_ID["C"], TOKEN_TO_ID["Q"]]) == "CQ"
