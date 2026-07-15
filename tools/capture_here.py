"""One-shot operator-driven capture: record the frequency YOU are tuned to, then
label it from the RBN spot log.

Flow (operator picks the signal; we never move the dial):
  1. read Lyra's current VFO over TCI
  2. record --dwell seconds of the audio you're listening to
  3. estimate the on-air signal freq from the dial + CW pitch, match it against
     recent spots in runs/rbn_spots_live.jsonl (written by rbn_spot_logger.py)
  4. decode the clip with the deployed model for a live sanity read
  5. save a Lyra-shaped clip (session.json + audio.wav) with the RBN callsign as
     the ground-truth label, into --out (default runs/rbn_cwt)

Run with DEEPFIST_CONDITION=1 (the deployed model is conditioned):
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/capture_here.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rbn_harvest import read_vfo, _capture, write_sample, is_keyed_signal, Candidate


def signal_khz_from_dial(dial_hz: int, sideband: str, pitch: float) -> float:
    """Invert rbn_harvest.dial_hz_for: recover the on-air signal freq from the dial."""
    sb = sideband.upper()
    if sb.endswith("L") or sb == "LSB":
        return (dial_hz - pitch) / 1000.0
    return (dial_hz + pitch) / 1000.0


def match_rbn(spots_path: Path, signal_khz: float, tol_khz: float, window_s: float):
    """Return (best_candidate|None, nearby_list) from the live spot log around signal_khz."""
    if not spots_path.exists():
        return None, []
    now = time.time()
    by_call: dict[str, dict] = {}
    for line in spots_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if now - s["t"] > window_s:
            continue
        if abs(s["freq_khz"] - signal_khz) > tol_khz:
            continue
        c = by_call.setdefault(s["call"], {"spotters": set(), "snr": -99, "wpm": [],
                                           "freqs": [], "last": 0.0})
        c["spotters"].add(s["spotter"])
        c["snr"] = max(c["snr"], s["snr"])
        c["wpm"].append(s["wpm"])
        c["freqs"].append(s["freq_khz"])
        c["last"] = max(c["last"], s["t"])
    ranked = sorted(by_call.items(), key=lambda kv: (len(kv[1]["spotters"]), kv[1]["snr"]),
                    reverse=True)
    nearby = [(call, len(d["spotters"]), d["snr"],
               sorted(d["freqs"])[len(d["freqs"]) // 2], int(now - d["last"]))
              for call, d in ranked]
    if not ranked:
        return None, nearby
    call, d = ranked[0]
    wpms = sorted(d["wpm"])
    cand = Candidate(call=call, freq_khz=sorted(d["freqs"])[len(d["freqs"]) // 2],
                     skimmers=len(d["spotters"]), peak_snr=d["snr"],
                     wpm=wpms[len(wpms) // 2], spotters=sorted(d["spotters"]))
    return cand, nearby


def deepfist_decode(ckpt: Path, sr: int, audio) -> str:
    """Readable greedy decode of the captured clip with the deployed model
    (conditioned when DEEPFIST_CONDITION=1 — matches how it was trained)."""
    import torch
    from scipy.signal import resample_poly
    from deepfist.model.net import CwCtcNet
    from deepfist.model.decode import greedy_ctc_decode
    from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
    from deepfist.features.conditioner import maybe_condition
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    net = CwCtcNet(time_downsample=cfg["time_downsample"], width=cfg["width"])
    net.load_state_dict(torch.load(str(ckpt), map_location="cpu")); net.eval()
    a = np.asarray(audio, dtype=np.float32)
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    with torch.no_grad():
        lp = net(audio_to_spectrogram(maybe_condition(a, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
    return greedy_ctc_decode(lp)[0]


async def main_async(args):
    dial = await read_vfo(args.uri, args.rx)
    if dial is None:
        print("ERROR: could not read VFO from TCI — is Lyra running with TCI on?", flush=True)
        return 2
    sig_khz = signal_khz_from_dial(dial, args.sideband, args.pitch)
    print(f"[vfo ] dial {dial} Hz  ->  signal ~{sig_khz:.2f} kHz ({args.sideband}, pitch {args.pitch:.0f})", flush=True)
    print(f"[rec ] capturing {args.dwell:.0f}s ...", flush=True)
    sr, audio = await _capture(args.uri, args.dwell, args.rx)
    keyed = is_keyed_signal(audio, sr, args.key_cov)
    peak = float(np.abs(audio).max())
    print(f"[rec ] got {len(audio)/sr:.1f}s @ {sr}Hz  peak={peak:.3f}  keyed_CW={'yes' if keyed else 'NO (weak/quiet?)'}", flush=True)

    cand, nearby = match_rbn(Path(args.spots), sig_khz, args.tol, args.window)
    if nearby:
        print("[rbn ] nearby spots (call, skimmers, snr, kHz, age_s):", flush=True)
        for c, n, snr, fk, age in nearby[:5]:
            print(f"         {c:8s} {n:2d} skim  snr{snr:>3}  {fk:.2f}  {age}s ago", flush=True)
    else:
        print(f"[rbn ] no spots within +/-{args.tol}kHz of {sig_khz:.2f} in last {args.window:.0f}s", flush=True)

    ours = deepfist_decode(Path(args.ckpt), sr, audio)
    print(f"[dcod] {Path(args.ckpt).parent.name}: {ours!r}", flush=True)

    label_cand = cand or Candidate(call="UNKNOWN", freq_khz=sig_khz, skimmers=0,
                                   peak_snr=0, wpm=0, spotters=[])
    agree = bool(cand) and cand.call in ours.replace(" ", "")
    d = write_sample(args.out, label_cand, args.sideband, dial, sr, audio,
                     our_pick=(cand.call if agree else None), margin=None, agree=agree)
    rbn_txt = f"{cand.call} ({cand.skimmers} skim, snr{cand.peak_snr})" if cand else "UNKNOWN"
    print(f"[save] RBN={rbn_txt}  agree={'YES' if agree else 'no'}  -> {d.name}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--uri", default="ws://127.0.0.1:40001")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=18.0)
    ap.add_argument("--pitch", type=float, default=650.0)
    ap.add_argument("--sideband", default="CWU")
    ap.add_argument("--spots", default="runs/rbn_spots_live.jsonl")
    ap.add_argument("--tol", type=float, default=0.4, help="freq match tolerance kHz")
    ap.add_argument("--window", type=float, default=200.0, help="max spot age seconds")
    ap.add_argument("--key-cov", type=float, default=0.65)
    ap.add_argument("--ckpt", default="runs/exp27_bt/model.pt")
    ap.add_argument("--out", default="runs/rbn_cwt")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
