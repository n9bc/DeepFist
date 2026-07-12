"""Real-time streaming CW decode over TCI (Thetis / ExpertSDR3).

Self-contained raw-WebSocket TCI client: connects, waits for READY, requests
RX audio (48 kHz stereo -> 8 kHz mono). Runs the DeepFist model on a rolling
window several times per second and STREAMS characters out as they stabilize
(~1 s behind live), instead of dumping a whole window every N seconds.

Decode-time tricks that make the live copy readable:
  * activity gating -- an independent in-band CW detector; silence never prints.
  * blank penalty   -- counter the model's CTC blank over-prediction on real audio.
  * frame-timed commit -- each decoded char has a frame time; a char is printed
    once its audio is older than `guard` seconds (settled), so re-decodes of the
    overlapping window don't reprint it.

    .venv/Scripts/python.exe scripts/tci_decode.py --uri ws://127.0.0.1:40001
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
HDR = 64
TYPE_RX_AUDIO = 1


def parse_packet(buf):
    if len(buf) < HDR:
        return None
    rx, sr, _fmt, _codec, _crc, length, dtype, ch = struct.unpack_from("<8I", buf)
    samples = np.frombuffer(buf[HDR:HDR + length * 4], dtype="<f4")
    return dtype, rx, sr, max(ch, 1), samples


def greedy_frames(log_probs, blank_pen):
    """Return (char_ids, frame_indices, n_frames) with a blank-logit penalty."""
    lp = log_probs.clone()
    lp[..., BLANK_ID] -= blank_pen
    args = lp.argmax(-1)[:, 0].tolist()
    prev, ids, frames = None, [], []
    for t, s in enumerate(args):
        if s != prev:
            if s != BLANK_ID:
                ids.append(s)
                frames.append(t)
            prev = s
    return ids, frames, len(args)


def cw_activity(a, sr):
    """Independent detector: keyed CW present in this window? Returns (bool, pitch)."""
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


async def run(uri, ckpt, rx_target, window, tick, guard, seconds, blank_pen):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=2).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    state = {"sr": None, "factor": 1, "freq": None, "total": 0}
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
                    print(f"RX {sr} Hz x{ch} -> {MODEL_SR} Hz | model={Path(ckpt).parent.name} "
                          f"pen={blank_pen} window={window}s tick={tick}s\n", flush=True)
                n = len(mono)
                state["total"] += n
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
        print(f"streaming RX{rx_target}  (~{guard:.1f}s latency, updates {1/tick:.0f}x/s)\n>> ",
              end="", flush=True)

        committed_t = 0.0     # absolute audio time already printed
        idle_gap = False
        start = time.time()
        try:
            while True:
                await asyncio.sleep(tick)
                if ring is None or state["total"] == 0:
                    continue
                audio_end = state["total"] / state["sr"]
                buf = ring.copy()
                audio = decimate(buf, state["factor"], ftype="fir").astype(np.float32) \
                    if state["factor"] > 1 else buf
                active, _pitch = cw_activity(audio, MODEL_SR)
                if not active:
                    if not idle_gap:
                        sys.stdout.write(" "); sys.stdout.flush(); idle_gap = True
                    committed_t = max(committed_t, audio_end - guard)
                    if seconds and time.time() - start >= seconds:
                        break
                    continue
                idle_gap = False
                with torch.no_grad():
                    spec = audio_to_spectrogram(audio, MODEL_SR).unsqueeze(0).unsqueeze(0).to(device)
                    ids, frames, T = greedy_frames(net(spec), blank_pen)
                # Commit the newly-settled time band (committed_t, audio_end-guard]
                # exactly once. Advancing the boundary by settled time (not by the
                # last char's drifting timestamp) prevents re-emitting boundary chars.
                win_start = audio_end - window
                settle_to = audio_end - guard
                emit = [cid for cid, fr in zip(ids, frames)
                        if committed_t < win_start + (fr / max(1, T)) * window <= settle_to]
                committed_t = max(committed_t, settle_to)
                if emit:
                    sys.stdout.write(ids_to_text(emit)); sys.stdout.flush()
                if seconds and time.time() - start >= seconds:
                    break
        finally:
            print(flush=True)
            try:
                await ws.send(f"AUDIO_STOP:{rx_target};")
            except Exception:
                pass
            recv_task.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:40001")
    ap.add_argument("--ckpt", default="runs/exp2/model_8000.pt")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--window", type=float, default=6.0, help="model context window (s)")
    ap.add_argument("--tick", type=float, default=0.4, help="decode interval (s)")
    ap.add_argument("--guard", type=float, default=1.3, help="commit delay / latency (s)")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--blank-penalty", type=float, default=3.0, dest="blank_pen")
    args = ap.parse_args()
    asyncio.run(run(args.uri, args.ckpt, args.rx, args.window, args.tick, args.guard,
                    args.seconds, args.blank_pen))


if __name__ == "__main__":
    main()
