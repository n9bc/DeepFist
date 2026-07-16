"""Real-time streaming CW decode over TCI. Default target: Lyra / ExpertSDR3
on ws://127.0.0.1:40001 (also works with Thetis, pass --uri).

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import ids_to_text, BLANK_ID
from squelch import has_signal, DEFAULT_THRESH
from despike import despike as _despike

MODEL_SR = SAMPLE_RATE          # 3200 Hz — must match the trained model
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


async def run(uri, ckpt, rx_target, window, tick, guard, seconds, blank_pen, squelch, despike,
              strip_punct=True, reconnect=True):
    """Load the model once, then stream — auto-reconnecting when Lyra drops
    (it crashes ~every 15 min; the decoder should survive, not die with it)."""
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_path = Path(ckpt).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    net = CwCtcNet(time_downsample=cfg.get("time_downsample", 2),
                   width=cfg.get("width", 1.0)).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    deadline = (time.time() + seconds) if seconds else None
    while True:
        try:
            await _stream_session(uri, net, device, ckpt, rx_target, window, tick, guard,
                                  deadline, blank_pen, squelch, despike, strip_punct)
            return                                          # deadline reached / normal end
        except (websockets.exceptions.WebSocketException, OSError, ConnectionError) as e:
            if not reconnect:
                raise
            if deadline and time.time() >= deadline:
                return
            print(f"\n[TCI lost ({type(e).__name__}) — reconnecting in 3s]", flush=True)
            await asyncio.sleep(3.0)


async def _stream_session(uri, net, device, ckpt, rx_target, window, tick, guard,
                          deadline, blank_pen, squelch, despike, strip_punct):
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
                          f"pen={blank_pen} squelch={squelch} window={window}s tick={tick}s\n",
                          flush=True)
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
        try:
            while True:
                await asyncio.sleep(tick)
                if deadline and time.time() >= deadline:
                    break
                if ring is None or state["total"] == 0:
                    continue
                audio_end = state["total"] / state["sr"]
                buf = ring.copy()
                audio = decimate(buf, state["factor"], ftype="fir").astype(np.float32) \
                    if state["factor"] > 1 else buf
                active, _score = has_signal(audio, MODEL_SR, squelch)   # keying gate on RAW audio
                if not active:
                    if not idle_gap:
                        sys.stdout.write(" "); sys.stdout.flush(); idle_gap = True
                    committed_t = max(committed_t, audio_end - guard)
                    if deadline and time.time() >= deadline:
                        break
                    continue
                idle_gap = False
                with torch.no_grad():
                    clean = _despike(audio, MODEL_SR) if despike else audio  # impulse-blank crashes
                    cond = condition(clean, MODEL_SR)   # AGC + tone-lock + bandpass — model expects this
                    spec = audio_to_spectrogram(cond, MODEL_SR).unsqueeze(0).unsqueeze(0).to(device)
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
                    text = ids_to_text(emit)
                    if strip_punct:
                        # exp16's , and . are ~always spurious gap-noise on real audio
                        # (ARRL CER 9.7->6.7 / 11.7->9.4 / 5.9->3.7 just by dropping them)
                        text = text.replace(",", "").replace(".", "")
                    if text:
                        sys.stdout.write(text); sys.stdout.flush()
                if deadline and time.time() >= deadline:
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
    ap.add_argument("--ckpt", default="runs/exp27_bt/model.pt")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--window", type=float, default=6.0, help="model context window (s)")
    ap.add_argument("--tick", type=float, default=0.4, help="decode interval (s)")
    ap.add_argument("--guard", type=float, default=1.3, help="commit delay / latency (s)")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--blank-penalty", type=float, default=0.0, dest="blank_pen",
                    help="0 with conditioning (the old 3.0 compensated for missing conditioning)")
    ap.add_argument("--squelch", type=float, default=DEFAULT_THRESH,
                    help="keying-ratio gate; windows below this print nothing (no signal)")
    ap.add_argument("--no-despike", action="store_false", dest="despike",
                    help="disable impulse-noise blanking (on by default; neutral on clean audio)")
    ap.add_argument("--keep-punct", action="store_false", dest="strip_punct",
                    help="keep , and . in output (suppressed by default: ~always spurious)")
    ap.add_argument("--no-reconnect", action="store_false", dest="reconnect",
                    help="die when TCI drops instead of auto-reconnecting (Lyra is flaky)")
    args = ap.parse_args()
    asyncio.run(run(args.uri, args.ckpt, args.rx, args.window, args.tick, args.guard,
                    args.seconds, args.blank_pen, args.squelch, args.despike, args.strip_punct,
                    args.reconnect))


if __name__ == "__main__":
    main()
