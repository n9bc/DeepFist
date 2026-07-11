import torch
from deepfist.train.loop import overfit_tiny


def test_overfits_tiny_batch():
    # Wiring proof: must drive a clean tiny batch to CER 0.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = 1000 if device == "cuda" else 400   # keep CPU CI bounded
    loss, c, used = overfit_tiny(n_clips=6, max_steps=steps, device=device)
    assert loss < 0.05
    assert c == 0.0
