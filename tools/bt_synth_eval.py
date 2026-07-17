"""BT synthetic gate (promoted from scratch_bt_test.py): synthesize known text
(clean) at several speeds and check the model reads =, ==, + patterns.
Clean audio => separates PATTERN coverage from signal quality.

  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/bt_synth_eval.py [ckpt]

Standing adoption gate with tools/bt_real_eval.py (HANDOFF §18.30).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from deepfist.synth.generator import _render_cw
from deepfist.features.conditioner import condition
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode

SR = 3200
dev = "cpu"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
cfg = json.loads((Path(ckpt).parent / "config.json").read_text())
print(f"ckpt={ckpt}  width={cfg.get('width', 1.0)}")
net = CwCtcNet(width=cfg.get("width", 1.0),
               time_downsample=cfg.get("time_downsample", 2)).to(dev)
net.load_state_dict(torch.load(ckpt, map_location=dev))
net.eval()

def decode(text, wpm, seed=0):
    rng = np.random.default_rng(seed)
    n = int(9 * SR)                      # 9 s clip, plenty of room
    clip = _render_cw(rng, text, wpm, pitch=650.0, sr=SR, n=n, drift_max=0.0)
    cond = condition(clip, sr=SR)
    spec = audio_to_spectrogram(cond, sample_rate=SR).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        logp = net(spec)                 # [T, B=1, C] log-probs
    return greedy_ctc_decode(logp)[0]    # first (only) batch string

tests = [
    ("TEST R TEST", "single-char control"),
    ("TEST = TEST", "single BT (well covered)"),
    ("TEST == TEST", "double BT (0 in training)"),
    ("N9BC == 0123456789", "operator's exact pattern"),
    ("TEST + TEST", "single AR (0 in training)"),
]
for wpm in (15, 20, 25, 30):
    print(f"\n===== {wpm} WPM =====")
    for text, note in tests:
        # average 3 seeds (fist jitter varies) to see consistency
        outs = [decode(text, wpm, seed=s) for s in range(3)]
        print(f"  {text:22s} -> {outs}   [{note}]")
