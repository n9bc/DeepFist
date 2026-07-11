"""Greedy CTC decoding: log-probs -> text."""
import torch
from deepfist.morse.alphabet import TOKENS

BLANK_ID = 0


def ids_to_text(ids) -> str:
    return "".join(TOKENS[int(i)] for i in ids)


def greedy_ctc_decode(log_probs: torch.Tensor) -> list[str]:
    # log_probs: [T, B, C]
    args = log_probs.argmax(dim=-1)  # [T, B]
    T, B = args.shape
    out = []
    for b in range(B):
        prev = None
        collapsed = []
        for t in range(T):
            s = int(args[t, b])
            if s != prev:
                if s != BLANK_ID:
                    collapsed.append(s)
                prev = s
        out.append(ids_to_text(collapsed))
    return out
