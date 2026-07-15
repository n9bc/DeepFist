"""Hands-free operator-driven harvest: record on every VFO change.

Polls Lyra's VFO over TCI. Each time the operator tunes to a NEW frequency and
lets it settle, this records --dwell seconds of that signal, matches the dial
against the live RBN spot log (rbn_spot_logger.py) for a ground-truth callsign,
decodes with the deployed model for a live read, and saves a Lyra-shaped clip.
Never moves the dial — the operator drives; we just follow and capture.

  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/follow_vfo_capture.py \
      --dwell 10 --out runs/rbn_cwt_op
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rbn_harvest import read_vfo, _capture, write_sample, is_keyed_signal, Candidate
from tools.capture_here import signal_khz_from_dial, match_rbn, deepfist_decode


async def capture_and_save(args, dial: int):
    sig_khz = signal_khz_from_dial(dial, args.sideband, args.pitch)
    print(f"[tune] dial {dial} Hz -> signal ~{sig_khz:.2f} kHz; recording {args.dwell:.0f}s...", flush=True)
    sr, audio = await _capture(args.uri, args.dwell, args.rx)
    keyed = is_keyed_signal(audio, sr, args.key_cov)
    cand, nearby = match_rbn(Path(args.spots), sig_khz, args.tol, args.window)
    ours = deepfist_decode(Path(args.ckpt), sr, audio)
    label_cand = cand or Candidate(call="UNKNOWN", freq_khz=sig_khz, skimmers=0,
                                   peak_snr=0, wpm=0, spotters=[])
    agree = bool(cand) and cand.call in ours.replace(" ", "")
    d = write_sample(args.out, label_cand, args.sideband, dial, sr, audio,
                     our_pick=(cand.call if agree else None), margin=None, agree=agree)
    rbn = f"{cand.call}({cand.skimmers}sk snr{cand.peak_snr})" if cand else "UNKNOWN"
    flag = "" if keyed else " [NOT keyed-CW]"
    print(f"[save] RBN={rbn} agree={'Y' if agree else 'n'}{flag}  ours={ours!r}  -> {d.name}", flush=True)
    return d


async def run(args):
    dial0 = await read_vfo(args.uri, args.rx, timeout=2.0)
    if dial0 is None:
        print("ERROR: no VFO from TCI — is Lyra running with TCI on?", flush=True)
        return 2
    print(f"[follow] watching VFO (start {dial0} Hz). Tune to a station; each new "
          f"settled freq -> {args.dwell:.0f}s capture. Ctrl-C to stop.", flush=True)
    last_captured = 0
    stable_freq = dial0
    stable_since = time.time()
    while True:
        try:
            cur = await read_vfo(args.uri, args.rx, timeout=1.5)
        except Exception as e:  # noqa: BLE001 -- TCI blip; keep following
            print(f"[follow] TCI read failed ({e.__class__.__name__}); retry", flush=True)
            await asyncio.sleep(1.0)
            continue
        now = time.time()
        if cur is None:
            await asyncio.sleep(args.poll)
            continue
        if abs(cur - stable_freq) > args.settle_tol:
            stable_freq = cur
            stable_since = now                        # dial is moving -> reset settle timer
        elif (now - stable_since) >= args.settle_s and abs(stable_freq - last_captured) > args.move_thresh:
            try:
                await capture_and_save(args, stable_freq)
            except Exception as e:  # noqa: BLE001 -- one bad capture shouldn't kill the loop
                print(f"[follow] capture failed: {e.__class__.__name__}: {e}", flush=True)
            last_captured = stable_freq
        await asyncio.sleep(args.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--uri", default="ws://127.0.0.1:40001")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=10.0)
    ap.add_argument("--pitch", type=float, default=650.0)
    ap.add_argument("--sideband", default="CWU")
    ap.add_argument("--spots", default="runs/rbn_spots_live.jsonl")
    ap.add_argument("--tol", type=float, default=0.4, help="freq match tolerance kHz")
    ap.add_argument("--window", type=float, default=200.0, help="max spot age seconds")
    ap.add_argument("--key-cov", type=float, default=0.65)
    ap.add_argument("--ckpt", default="runs/exp27_bt/model.pt")
    ap.add_argument("--out", default="runs/rbn_cwt_op")
    ap.add_argument("--poll", type=float, default=0.6, help="VFO poll interval s")
    ap.add_argument("--settle-s", type=float, default=1.2, dest="settle_s",
                    help="hold time before a new freq counts as settled")
    ap.add_argument("--settle-tol", type=float, default=60.0, dest="settle_tol",
                    help="Hz jitter tolerated within a 'settled' dial")
    ap.add_argument("--move-thresh", type=float, default=150.0, dest="move_thresh",
                    help="Hz the dial must move from the last capture to arm a new one")
    args = ap.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\n[follow] stopped.", flush=True)


if __name__ == "__main__":
    main()
