"""Live CW decode over TCI (Thetis / ExpertSDR3) — demo-ready.

Self-contained raw-WebSocket TCI client: connects, waits for READY, requests
RX audio, and runs the trained DeepFist model on it. Tolerant of unknown TCI 2.0
`*_ex` commands (the eesdr_tci library crashes on those). Handles the server's
native rate/channels (48 kHz stereo) -> 8 kHz mono. Inference only.

Two decode-time tricks make the live copy readable:
  * activity gating  -- only decode windows that actually contain keyed CW
    (independent in-band detector), so silence/noise never prints garbage.
  * blank penalty     -- subtract a bias from the CTC blank logit to counter the
    model's blank over-prediction on real audio (recovers suppressed content).

    .venv/Scripts/python.exe scripts/tci_decode.py --uri ws://127.0.0.1:50001
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
from deepfist.model.decode import ids_to_text, BLANK_ID

MODEL_SR = 8000
HDR = 64          # TCI data packet header: 8x uint32 + 32 reserved bytes
TYPE_RX_AUDIO = 1


def parse_packet(buf):
    if len(buf) < HDR:
        return None
    rx, sr, _fmt, _codec, _crc, length, dtype, ch = struct.unpack_from("<8I", buf)
    samples = np.frombuffer(buf[HDR:HDR + length * 4], dtype="<f4")
    return dtype, rx, sr, max(ch, 1), samples


def greedy_penalized(log_probs, blank_pen):
    """Greedy CTC collapse with a penalty subtracted from the blank logit."""
    lp = log_probs.clone()
    lp[..., BLANK_ID] -= blank_pen
    args = lp.argmax(-1)[:, 0].tolist()
    prev, out = None, []
    for s in args:
        if s != prev:
            if s != BLANK_ID:
                out.append(s)
            prev = s
    return ids_to_text(out)


def cw_activity(a, sr):
    """Independent detector: is there keyed CW in this window? Returns (bool, pitch).

    Coherent envelope at the dominant in-band pitch; require on/off keying
    structure with adequate in-band SNR. Robust to AGC (level-independent).
    """
    w = np.hanning(len(a))
    S = np.abs(np.fft.rfft(a * w)); f = np.fft.rfftfreq(len(a), 1.0 / sr)
    m = (f >= 300) & (f <= 1000)
    if not m.any():
        return False, 0.0
    pitch = float(f[m][np.argmax(S[m])])
    t = np.arange(len(a)) / sr
    k = max(1, int(0.004 * sr)); kern = np.ones(k) / k
    i = np.convolve(a * np.cos(2 * np.pi * pitch * t), kern, "same")
    q = np.convolve(a * np.sin(2 * np.pi * pitch * t), kern, "same")
    env = 2 * np.sqrt(i * i + q * q)
    if env.max() <= 0:
        return False, pitch
    e = env / env.max()
    on, off = e > 0.5, e < 0.15
    if on.sum() < 0.03 * len(e) or off.sum() < 0.10 * len(e):
        return False, pitch
    snr = 20 * np.log10((np.median(e[on]) + 1e-9) / (np.median(e[off]) + 1e-9))
    return (snr > 6.0 and 0.03 < on.mean() < 0.9), pitch


async def run(uri, ckpt, rx_target, window, hop, seconds, blank_pen):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=2).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    state = {"sr": None, "factor": 1, "freq": None, "packets": 0}
    ring = None

    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        ready = False

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
                    print(f"RX audio {sr} Hz x{ch} -> {MODEL_SR} Hz | model={Path(ckpt).parent.name} "
                          f"blank_pen={blank_pen}\n", flush=True)
                state["packets"] += 1
                n = len(mono)
                if n >= len(ring):
                    ring[:] = mono[-len(ring):]
                else:
                    ring[:-n] = ring[n:]
                    ring[-n:] = mono

        recv_task = asyncio.create_task(receiver())
        recv_task.add_done_callback(lambda t: t.cancelled() or t.exception())

        t0 = time.time()
        while not ready and time.time() - t0 < 5:
            await asyncio.sleep(0.1)
        await ws.send("AUDIO_STREAM_SAMPLE_TYPE:float32;")
        await ws.send(f"AUDIO_START:{rx_target};")
        print(f"listening on RX{rx_target}  (window {window}s, gated + blank-penalized)\n", flush=True)

        start = time.time()
        try:
            while True:
                await asyncio.sleep(hop)
                if ring is None or state["packets"] == 0:
                    continue
                buf = ring.copy()
                audio = decimate(buf, state["factor"], ftype="fir").astype(np.float32) \
                    if state["factor"] > 1 else buf
                ts = time.strftime("%H:%M:%S")
                active, pitch = cw_activity(audio, MODEL_SR)
                if not active:
                    print(f"[{ts}]  · · ·  (no CW)", flush=True)
                else:
                    with torch.no_grad():
                        spec = audio_to_spectrogram(audio, MODEL_SR).unsqueeze(0).unsqueeze(0).to(device)
                        text = greedy_penalized(net(spec), blank_pen)
                    fk = f"{state['freq']/1000:.1f}kHz " if state["freq"] else ""
                    print(f"[{ts}] {fk}{pitch:.0f}Hz | {text}", flush=True)
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
    ap.add_argument("--ckpt", default="runs/exp2/model_8000.pt")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--hop", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--blank-penalty", type=float, default=3.0, dest="blank_pen")
    args = ap.parse_args()
    asyncio.run(run(args.uri, args.ckpt, args.rx, args.window, args.hop, args.seconds, args.blank_pen))


if __name__ == "__main__":
    main()
