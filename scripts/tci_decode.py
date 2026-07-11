"""Live CW decode over TCI (Thetis / ExpertSDR3).

Self-contained raw-WebSocket TCI client: connects, waits for READY, requests
RX audio, and runs the trained DeepFist model on it. Tolerant of unknown TCI 2.0
`*_ex` commands (the eesdr_tci library crashes on those). Handles the server's
native rate/channels (e.g. 48 kHz stereo) by deinterleaving to mono and
decimating to the model's 8 kHz. Inference only (live audio is unlabeled).

    .venv/Scripts/python.exe scripts/tci_decode.py --uri ws://127.0.0.1:50001 --seconds 60
"""
import argparse
import asyncio
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import websockets
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode

MODEL_SR = 8000
HDR = 64          # TCI data packet header: 8x uint32 + 32 reserved bytes
TYPE_RX_AUDIO = 1


def parse_packet(buf):
    """Return (data_type, rx, sample_rate, channels, float32_samples) or None."""
    if len(buf) < HDR:
        return None
    rx, sr, _fmt, _codec, _crc, length, dtype, ch = struct.unpack_from("<8I", buf)
    samples = np.frombuffer(buf[HDR:HDR + length * 4], dtype="<f4")
    return dtype, rx, sr, max(ch, 1), samples


async def run(uri, ckpt, rx_target, window, hop, seconds, gate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=2).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    state = {"sr": None, "factor": 1, "freq": None, "packets": 0}
    ring = None  # allocated once we learn the native sample rate

    async with websockets.connect(uri, max_size=None) as ws:
        ready = False
        # --- background receiver: fill the ring buffer from RX audio frames ---
        async def receiver():
            nonlocal ring, ready
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    for cmd in msg.strip().split(";"):
                        low = cmd.lower()
                        if low.startswith("ready"):
                            ready = True
                        elif low.startswith("vfo:"):
                            try:
                                state["freq"] = float(cmd.split(":", 1)[1].split(",")[-1])
                            except (ValueError, IndexError):
                                pass
                    continue
                pkt = parse_packet(msg)
                if not pkt:
                    continue
                dtype, rx, sr, ch, samples = pkt
                if dtype != TYPE_RX_AUDIO or rx != rx_target or samples.size == 0:
                    continue
                mono = samples.reshape(-1, ch).mean(axis=1) if ch > 1 else samples
                if ring is None:
                    state["sr"] = sr
                    state["factor"] = max(1, round(sr / MODEL_SR))
                    ring = np.zeros(int(window * sr), dtype=np.float32)
                    print(f"RX audio: {sr} Hz x{ch}ch -> mono -> /{state['factor']} -> {MODEL_SR} Hz",
                          flush=True)
                state["packets"] += 1
                n = len(mono)
                if n >= len(ring):
                    ring[:] = mono[-len(ring):]
                else:
                    ring[:-n] = ring[n:]
                    ring[-n:] = mono

        recv_task = asyncio.create_task(receiver())
        recv_task.add_done_callback(lambda t: t.cancelled() or t.exception())

        # wait for handshake, then request audio
        t0 = time.time()
        while not ready and time.time() - t0 < 5:
            await asyncio.sleep(0.1)
        print(f"connected to {uri} (ready={ready}); requesting RX{rx_target} audio", flush=True)
        await ws.send("AUDIO_STREAM_SAMPLE_TYPE:float32;")
        await ws.send(f"AUDIO_START:{rx_target};")
        print(f"decoding every {hop}s over {window}s windows\n", flush=True)

        start = time.time()
        try:
            while True:
                await asyncio.sleep(hop)
                ts = time.strftime("%H:%M:%S")
                if ring is None or state["packets"] == 0:
                    print(f"[{ts}] (no audio packets yet)", flush=True)
                else:
                    buf = ring.copy()
                    audio = decimate(buf, state["factor"], ftype="fir").astype(np.float32) \
                        if state["factor"] > 1 else buf
                    peak = float(np.abs(audio).max())
                    fk = f"{state['freq']/1000:.1f}kHz" if state["freq"] else "?"
                    if peak < gate:
                        print(f"[{ts}] {fk} (silence, peak={peak:.1e})", flush=True)
                    else:
                        with torch.no_grad():
                            spec = audio_to_spectrogram(audio, MODEL_SR).unsqueeze(0).unsqueeze(0).to(device)
                            text = greedy_ctc_decode(net(spec))[0]
                        print(f"[{ts}] {fk} peak={peak:.2f} | {text}", flush=True)
                if seconds and time.time() - start >= seconds:
                    break
        finally:
            try:
                await ws.send(f"AUDIO_STOP:{rx_target};")
            except Exception:
                pass
            recv_task.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:50001")
    ap.add_argument("--ckpt", default="runs/exp1/model.pt")
    ap.add_argument("--rx", type=int, default=0, help="TCI receiver index")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N s (0=forever)")
    ap.add_argument("--gate", type=float, default=1e-4, help="skip decode below this peak amplitude")
    args = ap.parse_args()
    asyncio.run(run(args.uri, args.ckpt, args.rx, args.window, args.hop, args.seconds, args.gate))


if __name__ == "__main__":
    main()
