"""Segment real RBN-labeled anchors into clean (audio, callsign) training pairs.

Each anchor is an 18 s QSO clip labeled with ONE RBN callsign, but the audio contains a
whole QSO -> audio != label. This localizes WHERE the callsign is actually sent: slide a
window, frame-timed greedy-decode each, and where the decode contains the target callsign,
extract that tight char-span (+pad) as a verified-clean pair. Output: data/realset_train/
(8 kHz wav + labels.jsonl). Run with DEEPFIST_CONDITION=1.
"""
from __future__ import annotations
import json
import sys
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import BLANK_ID, ids_to_text
from deepfist.morse.alphabet import TOKENS

WIN, HOP, PAD = 6.0, 0.4, 0.15   # WIN must match the model's TRAINED window (6 s); a 3 s
                                 # search window under-reads real calls and mislabels them
                                 # "no-locate" even when a 6 s window reads them clearly.
WINDOW = int(6.0 * SAMPLE_RATE)          # match the synthetic training window (6 s @ SAMPLE_RATE)
OUT = ROOT / "data" / "realset_train"


def load_net(ck="runs/exp16/model.pt"):
    cfg = json.loads((ROOT / ck).parent.joinpath("config.json").read_text())
    net = CwCtcNet(time_downsample=cfg.get("time_downsample", 2), width=cfg.get("width", 1.0))
    net.load_state_dict(torch.load(ROOT / ck, map_location="cpu")); net.eval()
    return net


def to8k(a, sr):
    a = a.astype(np.float32)
    if sr == SAMPLE_RATE:
        return a
    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(a, SAMPLE_RATE // g, sr // g).astype(np.float32)


def _frames(lp):
    args = lp.argmax(-1)[:, 0].tolist()
    prev, ids, frames = None, [], []
    for t, s in enumerate(args):
        if s != prev:
            if s != BLANK_ID:
                ids.append(s); frames.append(t)
            prev = s
    return ids, frames, len(args)


def _ed(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def find_span(seg8, net, call):
    """Locate `call` in this window by EDIT DISTANCE (the model reads real calls close but
    not exact). Return (t0,t1) sec of the best-matching char-span, or None if no span is
    within tolerance. The RBN call is the trusted label even when the model mis-reads it."""
    with torch.no_grad():
        x = maybe_condition(seg8, SAMPLE_RATE)
        lp = net(audio_to_spectrogram(x, SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
    ids, frames, T = _frames(lp)
    # drop SPACE tokens (callsigns have no internal space) but keep frame positions
    ch = [(TOKENS[i], f) for i, f in zip(ids, frames) if i != 1]
    if not ch or T == 0:
        return None
    L = len(call)
    tol = max(1, round(L * 0.34))          # ~1 error per 3 chars
    best = (tol + 1, None)
    for start in range(len(ch)):
        for wlen in (L - 1, L, L + 1):
            end = start + wlen
            if wlen < 1 or end > len(ch):
                continue
            cand = "".join(c for c, _ in ch[start:end])
            d = _ed(cand, call)
            if d < best[0]:
                best = (d, (ch[start][1], ch[end - 1][1]))
    if best[1] is None:
        return None
    dur = len(seg8) / SAMPLE_RATE
    return best[1][0] / T * dur, best[1][1] / T * dur


def segment_anchor(wav, call, net):
    sr, a = wavfile.read(str(wav))
    if a.ndim > 1:
        a = a.mean(1)
    a8 = to8k(a, sr)
    n = len(a8); dur = n / SAMPLE_RATE
    t = 0.0
    while t + WIN <= dur + HOP:
        seg = a8[int(t * SAMPLE_RATE):int(min(dur, t + WIN) * SAMPLE_RATE)]
        if len(seg) >= SAMPLE_RATE:
            span = find_span(seg, net, call)
            if span:
                lo = max(0.0, t + span[0] - PAD)
                hi = min(dur, t + span[1] + PAD)
                clip = a8[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
                # verify: the tight clip alone must decode to exactly the call
                if find_span(clip, net, call):
                    return clip
        t += HOP
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net = load_net()
    anchors = sorted((ROOT / "runs" / "rbn_anchors").glob("*/session.json"))
    rows, kept = [], 0
    for sj in anchors:
        m = json.loads(sj.read_text())
        call = m["rbn"]["callsign"].upper()
        wav = sj.parent / m["audio"][0]["file"]
        if not wav.exists():
            continue
        clip = segment_anchor(wav, call, net)
        status = "no-locate"
        if clip is not None and len(clip) >= int(0.4 * SAMPLE_RATE):
            # pad to the synthetic window length (silence) + save int16 so the WMR blend
            # loader (divides by 32768, stacks fixed-size) consumes it correctly.
            clip = clip[:WINDOW]
            padded = np.zeros(WINDOW, dtype=np.float32)
            padded[:len(clip)] = clip
            peak = float(np.abs(padded).max()) or 1.0
            clip16 = (padded / peak * 20000.0).astype(np.int16)
            fn = f"{call}_{kept:03d}.wav"
            wavfile.write(str(OUT / fn), SAMPLE_RATE, clip16)
            rows.append({"file": fn, "text": call, "src": sj.parent.name,
                         "dur": round(len(clip) / SAMPLE_RATE, 2),
                         "skimmers": m["rbn"]["skimmers"], "snr": m["rbn"]["peak_snr"]})
            kept += 1
            status = f"kept {rows[-1]['dur']}s"
        print(f"  {call:8} ({m['rbn']['skimmers']:2d}skim snr{m['rbn']['peak_snr']}) -> {status}",
              flush=True)
    (OUT / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nclean pairs: {kept}/{len(anchors)} anchors -> {OUT}")


if __name__ == "__main__":
    main()
