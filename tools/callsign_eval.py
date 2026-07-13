"""Callsign-copy eval: a scale proxy metric for the maximize-callsign-copy loop.

Builds a held-out set of clean/strong SINGLE-CALLSIGN clips (the target conditions) from
the synthetic generator, then reports exact-callsign accuracy for a checkpoint. Synthetic
=> distribution-leakage vs training; it is an iteration proxy, NOT the real-audio truth
(that stays the 3 operator/RBN-confirmed clips in tools/eval_realset.py). Cache under
data/callsign_eval/ so the set is FIXED across models. Run with DEEPFIST_CONDITION=1.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

EVAL_DIR = ROOT / "data" / "callsign_eval"
CALL_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z]{1,3}(/[0-9])?$")   # single callsign token


def build(n=150, seed0=900000):
    """Generate n clean/strong single-callsign clips (fixed, cached)."""
    import numpy as np
    from scipy.io import wavfile
    from deepfist.synth.generator import generate, GenConfig
    from deepfist.synth.channel import ChannelConfig
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig(snr_range=(8.0, 22.0), qrm_prob=0.1, channel=ChannelConfig())  # clean/strong
    rows, seed, kept = [], seed0, 0
    while kept < n:
        s = generate(seed=seed, config=cfg); seed += 1
        lbl = s.label.strip().upper()
        if not CALL_RE.match(lbl):
            continue
        fn = f"cs_{kept:04d}.wav"
        wavfile.write(str(EVAL_DIR / fn), 8000, np.asarray(s.audio, dtype=np.float32))
        rows.append({"file": fn, "call": lbl})
        kept += 1
    (EVAL_DIR / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def _load_net(ckpt):
    import torch
    from deepfist.model.net import CwCtcNet
    p = Path(ckpt) if Path(ckpt).is_absolute() else ROOT / ckpt
    cfg = {}
    cfgp = p.parent / "config.json"
    if cfgp.exists():
        cfg = json.loads(cfgp.read_text())
    net = CwCtcNet(time_downsample=cfg.get("time_downsample", 2), width=cfg.get("width", 1.0))
    net.load_state_dict(torch.load(p, map_location="cpu"))
    net.eval()
    return net


def evaluate(ckpt):
    """RAW greedy exact-match -- measures the model's intrinsic callsign copy, NOT the
    MASTER.SCP rescorer (which only fires for real calls and confounds a synthetic proxy)."""
    import numpy as np, torch
    from scipy.io import wavfile
    from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
    from deepfist.features.conditioner import maybe_condition
    from deepfist.model.decode import greedy_ctc_decode
    net = _load_net(ckpt)
    rows = [json.loads(l) for l in (EVAL_DIR / "labels.jsonl").read_text().splitlines() if l.strip()]
    ok = 0
    with torch.no_grad():
        for r in rows:
            _sr, a = wavfile.read(str(EVAL_DIR / r["file"]))
            x = maybe_condition(a.astype(np.float32), SAMPLE_RATE)
            lp = net(audio_to_spectrogram(x, SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            ok += (greedy_ctc_decode(lp)[0].strip() == r["call"])
    return ok, len(rows)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
    if not (EVAL_DIR / "labels.jsonl").exists():
        print(f"building callsign eval set..."); build()
    ok, n = evaluate(ckpt)
    print(f"{ckpt}: callsign-copy {ok}/{n} = {ok/n:.1%}  (synthetic proxy)")


if __name__ == "__main__":
    main()
