"""Real-time streaming decode DEMO from a recorded WAV.

Streams a recorded off-air capture through the exact live decode pipeline at
real-time pace, so it looks and behaves like the live decoder (character-by-
character, ~1s latency) but is reproducible for a demo. Same model, gating,
blank penalty, and frame-timed commit as scripts/tci_decode.py.

    .venv/Scripts/python.exe scripts/demo_file.py runs/live_capture.wav
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import ids_to_text
from scripts.tci_decode import greedy_frames, cw_activity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default="runs/exp2/model_8000.pt")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--tick", type=float, default=0.4)
    ap.add_argument("--guard", type=float, default=1.3)
    ap.add_argument("--blank-penalty", type=float, default=3.0, dest="pen")
    ap.add_argument("--realtime", action="store_true", help="pace playback in real time")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=2).to(dev); net.eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))

    sr, a = wavfile.read(args.wav)
    a = a.astype(np.float32)
    if abs(a).max() > 1.5:
        a /= 32768.0

    W = int(args.window * sr)
    print(f"streaming decode of {args.wav}  ({len(a)/sr:.0f}s @ {sr}Hz)\n>> ", end="", flush=True)
    committed = 0.0
    for end in range(int(args.tick * sr), len(a) + 1, int(args.tick * sr)):
        seg = a[max(0, end - W):end]
        if len(seg) < W:
            seg = np.concatenate([np.zeros(W - len(seg), np.float32), seg])
        audio_end = end / sr
        if not cw_activity(seg, sr)[0]:
            committed = max(committed, audio_end - args.guard)
        else:
            with torch.no_grad():
                spec = audio_to_spectrogram(seg, sr).unsqueeze(0).unsqueeze(0).to(dev)
                ids, frames, T = greedy_frames(net(spec), args.pen)
            win_start = audio_end - args.window
            settle_to = audio_end - args.guard
            emit = [c for c, fr in zip(ids, frames)
                    if committed < win_start + (fr / max(1, T)) * args.window <= settle_to]
            committed = max(committed, settle_to)
            if emit:
                sys.stdout.write(ids_to_text(emit)); sys.stdout.flush()
        if args.realtime:
            time.sleep(args.tick)
    print("\n")


if __name__ == "__main__":
    main()
