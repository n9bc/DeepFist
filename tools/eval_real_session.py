"""Honest real-audio eval: full-session windowed decode vs GROUND-TRUTH text.

Unlike the WMR/teacher benchmarks, labels here are ARRL's published transcript
(.txt) for each W1AW code-practice MP3 — real off-air-style audio with exact
ground truth. Decodes the whole session in non-overlapping windows with each
model, concatenates, and reports space-normalized CER against the cleaned
transcript. This is a *relative* yardstick (window-boundary loss hits all models
equally) but it is the first eval on real audio with true labels.

Usage:
  .venv/Scripts/python.exe tools/eval_real_session.py \
     --wav runs/real/arrl/arrl_20wpm_mono.wav --txt runs/real/arrl/arrl_20wpm.txt \
     --ckpt runs/exp11/model.pt --deepcw [--win 15 --norm-peak]
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.train.metrics import cer
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID

_b = Path(__file__).resolve().parent / "benchmark_vs_deepcw.py"
_sp = ilu.spec_from_file_location("bench", _b); _B = ilu.module_from_spec(_sp); _sp.loader.exec_module(_B)


def clean_transcript(raw: str) -> str:
    """ARRL .txt -> plain uppercase letters/digits/space reference string.

    Drops the '= ... =' / non-ASCII prosign+ID markers and station-ID lines;
    keeps only characters DeepFist can emit. Space-normalized downstream."""
    t = raw.upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)      # keep alnum + space only
    # remove the standard boilerplate tokens that appear as markers/announcements
    for junk in ["NOW", "WPM", "TEXT", "IS", "FROM", "QST", "PAGE", "DE", "W1AW", "END", "OF", "MAY", "JUNE", "JULY", "APRIL"]:
        pass  # keep them — W1AW actually SENDS these words on air; safer to keep
    t = re.sub(r"\s+", " ", t).strip()
    return t


def keep_tokenizable(s: str) -> str:
    out = []
    for ch in s:
        if ch == " " or ch in TOKEN_TO_ID:
            out.append(ch)
    return "".join(out)


def load_wav(path, norm_peak):
    sr, a = wavfile.read(str(path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
    if norm_peak:
        a = a / (np.abs(a).max() + 1e-9) * 0.95
    return sr, a


def windows(a, sr, win, hop):
    t = 0.0; dur = len(a) / sr
    while t + win <= dur + hop:  # include tail
        yield a[int(t*sr):int((t+win)*sr)]
        t += hop


def decode_ours(ckpt, a, sr, win, hop):
    cfg = json.loads((Path(ckpt).parent / "config.json").read_text())
    net = CwCtcNet(time_downsample=cfg["time_downsample"], width=cfg["width"])
    net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
    parts = []
    with torch.no_grad():
        for seg in windows(a, sr, win, hop):
            if len(seg) < sr * 1: continue
            x = resample_poly(seg, SAMPLE_RATE, sr).astype(np.float32)
            lp = net(audio_to_spectrogram(maybe_condition(x, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            parts.append(greedy_ctc_decode(lp)[0])
    return " ".join(parts)


def decode_deepcw(a, sr, win, hop):
    mod, meta, sess = _B.load_deepcw()
    tr = int(meta["sample_rate"]); chars = list(meta["chars"]); blank = int(meta["blank_index"])
    inn, outn = meta["onnx_input_name"], meta["onnx_output_name"]
    parts = []
    for seg in windows(a, sr, win, hop):
        if len(seg) < sr * 1: continue
        x = mod.resample_linear(seg.astype(np.float32), sr, tr)
        parts.append(mod.greedy_ctc_decode(sess.run([outn], {inn: mod.audio_to_spectrogram(x, meta)})[0], chars, blank))
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True); ap.add_argument("--txt", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--deepcw", action="store_true")
    ap.add_argument("--win", type=float, default=15.0); ap.add_argument("--hop", type=float, default=15.0)
    ap.add_argument("--norm-peak", action="store_true", help="peak-normalize whole clip to 0.95 first")
    args = ap.parse_args()

    ref = keep_tokenizable(clean_transcript(Path(args.txt).read_text(encoding="utf-8", errors="ignore")))
    sr, a = load_wav(args.wav, args.norm_peak)
    print(f"{Path(args.wav).name}  dur={len(a)/sr:.0f}s peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}  ref_len={len(ref.replace(' ',''))}")
    def scr(pred):
        return cer(pred.replace(" ", ""), ref.replace(" ", ""))
    for ckpt in args.ckpt:
        p = decode_ours(ckpt, a, sr, args.win, args.hop)
        print(f"  {Path(ckpt).parent.name:12s} CER {scr(p)*100:5.1f}%   {p[:70]!r}")
    if args.deepcw:
        p = decode_deepcw(a, sr, args.win, args.hop)
        print(f"  {'DeepCW':12s} CER {scr(p)*100:5.1f}%   {p[:70]!r}")
    print(f"  {'[ref]':12s}          {ref[:70]!r}")


if __name__ == "__main__":
    main()
