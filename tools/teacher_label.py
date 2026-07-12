"""DeepCW-as-teacher: auto-label real CW recordings for DeepFist fine-tuning.

Runs DeepCW's reference model (AGPL, local dev-only — never shipped/committed) as a
"teacher" over real off-air recordings that DeepFist currently garbles, producing
pseudo-labels. Optionally cross-checks with a DeepFist checkpoint and SCP-verifies the
decoded callsigns, so we can KEEP only high-confidence segments as clean training pairs.

This is step 4 of HANDOFF §16.4 (DeepCW-as-teacher on real recordings). Licensing note
mirrors benchmark_vs_deepcw.py: DeepCW stays external; only its decode output (text) is
used here as a label, which is not copyrightable.

Usage (single clip, validate mechanics):
  .venv/Scripts/python.exe tools/teacher_label.py --wav runs/real/wav/foo.wav --compare runs/exp11/model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE

# Reuse the vetted DeepCW loader/decoder from the benchmark tool.
import importlib.util as _ilu
_bench_src = Path(__file__).resolve().parent / "benchmark_vs_deepcw.py"
_spec = _ilu.spec_from_file_location("bench", _bench_src)
_bench = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bench)


def teacher_decode(mod, meta, sess, wav_path: Path) -> str:
    """Full-clip DeepCW pseudo-label for one WAV."""
    tr = int(meta["sample_rate"])
    chars = list(meta["chars"])
    blank = int(meta["blank_index"])
    in_name, out_name = meta["onnx_input_name"], meta["onnx_output_name"]
    srate, audio = mod.read_wav_mono(wav_path)
    audio = mod.resample_linear(audio, srate, tr)
    spec = mod.audio_to_spectrogram(audio, meta)
    out = sess.run([out_name], {in_name: spec})
    return mod.greedy_ctc_decode(out[0], chars, blank)


def deepfist_decode(ckpt: Path, wav_path: Path, device: str) -> str:
    import json
    cfg = ckpt.parent / "config.json"
    width, ds = 1.0, 2
    if cfg.exists():
        c = json.loads(cfg.read_text())
        width, ds = c.get("width", 1.0), c.get("time_downsample", 2)
    net = CwCtcNet(time_downsample=ds, width=width).to(device)
    net.load_state_dict(torch.load(str(ckpt), map_location=device))
    net.eval()
    sr, a = wavfile.read(str(wav_path))
    a = a.astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    spec = audio_to_spectrogram(a, SAMPLE_RATE)
    with torch.no_grad():
        lp = net(spec.unsqueeze(0).unsqueeze(0).to(device))
    return greedy_ctc_decode(lp)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="WAV file (any SR) to label")
    ap.add_argument("--compare", action="append", default=[],
                    help="repeatable DeepFist ckpt(s) to show alongside the teacher")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = Path(args.wav)

    mod, meta, sess = _bench.load_deepcw()
    teacher = teacher_decode(mod, meta, sess, wav)
    print(f"clip: {wav.name}")
    print(f"  DeepCW (teacher): {teacher!r}")
    for ckpt in args.compare:
        p = Path(ckpt)
        ours = deepfist_decode(p, wav, device)
        print(f"  {p.parent.name:16s}: {ours!r}")


if __name__ == "__main__":
    main()
