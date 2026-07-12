"""Capture live TCI RX audio (from Lyra/ExpertSDR3 or Thetis) to a WAV file.

Connects to a TCI server over a raw WebSocket (the `eesdr_tci` lib crashes on
Thetis 2.0 `*_ex` commands, so we speak the protocol directly), streams RX audio,
and writes it to a WAV at the radio's native sample rate (typically 48 kHz). Use it
to grab real off-air CW for offline decode/benchmarking through the conditioned
pipeline (tools/eval_real_session.py, tools/benchmark_vs_deepcw.py, teacher labeling).

TCI notes:
- Lyra/ExpertSDR3: ws://127.0.0.1:40001 (RX0 in CW). Thetis: ws://127.0.0.1:50001.
- RX audio arrives as 48 kHz float32; for a CW RX the 2 channels are duplicated
  (L==R, verified corr=1.0), so we average to mono safely.
- Lyra audio is high-scale (peak ~16-22, RMS ~5-7), NOT [-1,1]. That's expected —
  the front-end conditioner's AGC normalizes it. The WAV is saved at native scale so
  you can inspect the real level; conditioning/decoding handles normalization.

Usage:
  .venv/Scripts/python.exe scripts/tci_capture.py [--uri ws://127.0.0.1:40001] \
      [--seconds 18] [--rx 0] [--out runs/lyra_live.wav]

Then decode/compare, e.g.:
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/eval_real_session.py \
      --wav runs/lyra_live.wav --txt <transcript_if_any> --ckpt runs/exp15/model.pt --deepcw
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tci_decode import parse_packet, TYPE_RX_AUDIO


async def capture(uri: str, seconds: float, rx: int) -> tuple[int, np.ndarray]:
    chunks: list[np.ndarray] = []
    info = {"sr": None, "ch": None, "t0": None}
    ready = False

    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        async def reader():
            nonlocal ready
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    if any(c.lower().startswith("ready") for c in msg.split(";")):
                        ready = True
                    continue
                pkt = parse_packet(msg)
                if not pkt:
                    continue
                dtype, prx, sr, ch, samples = pkt
                if dtype != TYPE_RX_AUDIO or prx != rx or samples.size == 0:
                    continue
                if info["sr"] is None:
                    info["sr"], info["ch"], info["t0"] = sr, ch, time.time()
                    print(f"RX{rx} {sr} Hz x{ch} — capturing {seconds}s...", flush=True)
                mono = samples.reshape(-1, ch).mean(axis=1) if ch > 1 else samples
                chunks.append(mono.astype(np.float32))

        task = asyncio.create_task(reader())
        start = time.time()
        while not ready and time.time() - start < 5:
            await asyncio.sleep(0.1)
        await ws.send("AUDIO_STREAM_SAMPLE_TYPE:float32;")
        await ws.send(f"AUDIO_START:{rx};")
        while info["t0"] is None or (time.time() - info["t0"]) < seconds:
            await asyncio.sleep(0.2)
        task.cancel()

    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return info["sr"] or 48000, audio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default="ws://127.0.0.1:40001", help="TCI WebSocket URI")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--rx", type=int, default=0, help="TCI receiver index (RX0 = CW)")
    ap.add_argument("--out", default="runs/lyra_live.wav")
    args = ap.parse_args()

    sr, audio = asyncio.run(capture(args.uri, args.seconds, args.rx))
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    print(f"captured {len(audio)/sr:.1f}s @ {sr} Hz  peak={peak:.2f} rms={rms:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.out, sr, audio.astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
