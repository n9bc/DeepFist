"""Model-INDEPENDENT callsign localizer (matched filter / "fingerprint").

We already KNOW each anchor's callsign (RBN) and the speed it was sent (RBN wpm). A callsign
in CW is a fixed ON/OFF rhythm. So instead of asking the model to read it (which fails on the
hard ones), we synthesize the callsign's ideal keying envelope at the known wpm and slide it
against the recording's tone envelope (normalized cross-correlation). The best-correlating
offset is WHERE the callsign is sent — even when the model reads the audio as garbage.

Cut a 6 s window there, label it with the trusted RBN callsign -> clean training pair.

Positive control: on clips segment_anchors already located, this should score high too.

CLI:
  .venv/Scripts/python.exe tools/locate_callsign_template.py            # calibrate: score the pool
  .venv/Scripts/python.exe tools/locate_callsign_template.py --build --thresh 0.45  # write clips
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from deepfist.morse.alphabet import MORSE
from deepfist.features.spectrogram import SAMPLE_RATE

FRAME_MS = 5.0            # envelope frame period
OUT = ROOT / "data" / "realset_train_tmpl"
WINDOW = int(6.0 * SAMPLE_RATE)


def morse_template(call: str, wpm: float) -> np.ndarray:
    """0/1 keying envelope for `call` at `wpm`, sampled every FRAME_MS."""
    unit_ms = 1200.0 / wpm                       # PARIS dot length
    units = []                                   # (on?, n_units)
    chars = [c for c in call.upper() if c in MORSE]
    for ci, ch in enumerate(chars):
        elems = MORSE[ch]
        for ei, e in enumerate(elems):
            units.append((1, 3 if e == "-" else 1))     # dah=3, dit=1
            units.append((0, 1))                          # intra-char gap
        if ci != len(chars) - 1:
            units.append((0, 2))                          # -> 3-unit inter-char gap
    # render to frames
    seq = []
    for on, n in units:
        seq.extend([on] * max(1, int(round(n * unit_ms / FRAME_MS))))
    return np.asarray(seq, dtype=np.float32)


def tone_envelope(a: np.ndarray, sr: int) -> np.ndarray:
    """Recording's CW-tone amplitude over time (frames of FRAME_MS)."""
    a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    hop = max(1, int(sr * FRAME_MS / 1000))
    nper = max(hop * 2, 256)
    f, t, Z = stft(a, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None)
    mag = np.abs(Z)
    band = (f >= 250) & (f <= 1500)              # CW audio-tone range
    bidx = np.where(band)[0]
    sub = mag[bidx]
    peak = bidx[np.argmax(sub.mean(1))]          # sustained tone bin
    lo, hi = max(0, peak - 1), min(mag.shape[0], peak + 2)
    env = mag[lo:hi].sum(0)
    env = env / (env.max() + 1e-9)
    return env


def ncc(env: np.ndarray, templ: np.ndarray) -> tuple[float, int]:
    """Best normalized cross-correlation (Pearson r) of templ within env; (r, frame_offset)."""
    L = len(templ)
    if L >= len(env):
        return -1.0, 0
    t = templ - templ.mean()
    tnorm = float(np.sqrt((t * t).sum())) + 1e-9
    cross = np.correlate(env, t, mode="valid")
    c1 = np.cumsum(np.insert(env, 0, 0.0))
    c2 = np.cumsum(np.insert(env * env, 0, 0.0))
    wsum = c1[L:] - c1[:-L]
    wsq = c2[L:] - c2[:-L]
    wstd = np.sqrt(np.maximum(wsq - wsum * wsum / L, 1e-12))
    r = cross / (wstd * tnorm)
    k = int(np.argmax(r))
    return float(r[k]), k


def locate(a, sr, call, wpm):
    """Return (best_r, t0_sec, t1_sec) trying a few wpm scales around the RBN estimate."""
    env = tone_envelope(a, sr)
    best = (-1.0, 0.0, 0.0)
    for scale in (0.88, 0.94, 1.0, 1.06, 1.12):
        templ = morse_template(call, wpm * scale)
        r, k = ncc(env, templ)
        if r > best[0]:
            t0 = k * FRAME_MS / 1000.0
            t1 = (k + len(templ)) * FRAME_MS / 1000.0
            best = (r, t0, t1)
    return best


def to_sr(a, sr):
    from math import gcd
    from scipy.signal import resample_poly
    a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    if sr == SAMPLE_RATE:
        return a
    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(a, SAMPLE_RATE // g, sr // g).astype(np.float32)


def pool_sessions():
    names = (ROOT / "data" / "callsign_train_pool.txt").read_text().split()
    out = []
    for n in names:
        for d in ("rbn_anchors", "rbn_harvest"):
            sj = ROOT / "runs" / d / n / "session.json"
            if sj.exists():
                out.append(sj); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="write clips (else just calibrate/report)")
    ap.add_argument("--thresh", type=float, default=0.45, help="min correlation to accept")
    args = ap.parse_args()

    # positive-control set: calls segment_anchors already located (should score high)
    located = set()
    lp = ROOT / "data" / "realset_train_pool" / "labels.jsonl"
    if lp.exists():
        for l in lp.read_text().splitlines():
            if l.strip():
                located.add(Path(json.loads(l)["file"]).stem.rsplit("_", 1)[0])

    rows, scored = [], []
    for sj in pool_sessions():
        m = json.loads(sj.read_text())
        call = m["rbn"]["callsign"].upper()
        wpm = m["rbn"].get("wpm") or 20
        wav = sj.parent / m["audio"][0]["file"]
        if not wav.exists():
            continue
        sr, a = wavfile.read(str(wav))
        r, t0, t1 = locate(a, sr, call, wpm)
        scored.append((r, call, wpm, "ctrl" if call in located else "new", sj, t0, t1, wav, sr))

    scored.sort(reverse=True)
    ctrl = [s[0] for s in scored if s[3] == "ctrl"]
    print(f"pool clips scored: {len(scored)}   accept thresh r>{args.thresh}")
    if ctrl:
        print(f"positive-control (segment_anchors-located) r: min={min(ctrl):.2f} "
              f"median={np.median(ctrl):.2f} max={max(ctrl):.2f}  (n={len(ctrl)})")
    n_acc = sum(1 for s in scored if s[0] >= args.thresh)
    print(f"would accept: {n_acc}/{len(scored)} clips\n")
    print("  r     call      wpm  kind   span(s)")
    for r, call, wpm, kind, sj, t0, t1, wav, sr in scored[:28]:
        print(f"  {r:.2f}  {call:9} {wpm:>3}  {kind:4}  {t0:5.1f}-{t1:4.1f}")

    if not args.build:
        print("\n(dry run — pass --build --thresh <r> to write clips)")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    kept = 0
    for r, call, wpm, kind, sj, t0, t1, wav, sr in scored:
        if r < args.thresh:
            continue
        sr2, a = wavfile.read(str(wav))
        a8 = to_sr(a, sr2)
        c = (t0 + t1) / 2.0
        lo = max(0.0, c - 3.0); hi = lo + 6.0
        seg = a8[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
        buf = np.zeros(WINDOW, dtype=np.float32)
        buf[:len(seg)] = seg[:WINDOW]
        peak = float(np.abs(buf).max()) or 1.0
        fn = f"{call}_{kept:03d}.wav"
        wavfile.write(str(OUT / fn), SAMPLE_RATE, (buf / peak * 20000).astype(np.int16))
        rows.append({"file": str((OUT / fn).resolve()), "text": call, "r": round(r, 3)})
        kept += 1
    (OUT / "labels.jsonl").write_text(
        "\n".join(json.dumps({k: r[k] for k in ("file", "text")}) for r in rows) + "\n")
    print(f"\nwrote {kept} clips -> {OUT}")


if __name__ == "__main__":
    main()
