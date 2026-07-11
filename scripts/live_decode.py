"""Live CW decode: capture PC audio (e.g. Virtual Audio Cable from Thetis),
window it, and run the trained DeepFist model on the real off-air signal.

This validates the synthetic-trained model on real audio. It is inference only
(the live signal is unlabeled, so it cannot be used for supervised training).

Example:
    python scripts/live_decode.py --device "Line 3 (Virtual Audio Cable)" --ckpt runs/exp1/model.pt
"""
import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import decimate
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode


def find_device(spec: str):
    """Resolve a device by index (as a string) or case-insensitive name substring."""
    if spec.isdigit():
        idx = int(spec)
        return idx, sd.query_devices(idx)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i, d
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Line 3 (Virtual Audio Cable)")
    ap.add_argument("--ckpt", default="runs/exp1/model.pt")
    ap.add_argument("--window", type=float, default=6.0, help="seconds of audio per decode")
    ap.add_argument("--hop", type=float, default=3.0, help="seconds between decodes")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N s (0=until Ctrl+C)")
    ap.add_argument("--target-sr", type=int, default=8000)
    ap.add_argument("--gate", type=float, default=1e-4, help="skip decode below this peak amplitude")
    args = ap.parse_args()

    dev_idx, dev = find_device(args.device)
    if dev is None:
        print(f"No input device matching {args.device!r}. Run with --device <index>.")
        sys.exit(1)
    cap_sr = int(dev["default_samplerate"])
    channels = min(2, dev["max_input_channels"])
    factor = round(cap_sr / args.target_sr)
    print(f"capturing [{dev_idx}] {dev['name']} @ {cap_sr} Hz -> /{factor} -> {args.target_sr} Hz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=2).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    ring = np.zeros(int(args.window * cap_sr), dtype=np.float32)
    lock = threading.Lock()

    def callback(indata, frames, tinfo, status):
        mono = indata.mean(axis=1).astype(np.float32)
        n = len(mono)
        with lock:
            ring[:-n] = ring[n:]
            ring[-n:] = mono

    stream = sd.InputStream(device=dev_idx, channels=channels, samplerate=cap_sr,
                            dtype="float32", callback=callback)
    print("listening... (Ctrl+C to stop)\n")
    start = time.time()
    with stream:
        try:
            while True:
                time.sleep(args.hop)
                with lock:
                    buf = ring.copy()
                audio = decimate(buf, factor, ftype="fir").astype(np.float32) if factor > 1 else buf
                ts = time.strftime("%H:%M:%S")
                peak = float(np.abs(audio).max())
                if peak < args.gate:
                    print(f"[{ts}] (silence, peak={peak:.1e})")
                else:
                    with torch.no_grad():
                        spec = audio_to_spectrogram(audio, args.target_sr).unsqueeze(0).unsqueeze(0).to(device)
                        text = greedy_ctc_decode(net(spec))[0]
                    print(f"[{ts}] peak={peak:.2f} | {text}")
                if args.seconds and time.time() - start >= args.seconds:
                    break
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
