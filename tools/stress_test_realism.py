"""No-training synthetic realism stress test (HANDOFF §17.6, Codex diagnostic).

Goal: with the EXISTING deployed model (no retraining), find which missing
synthetic-realism factor most inflates space-normalized CER toward the ~5% real
off-air level. We build controlled high-speed CW clips (30/35/40 WPM), hold
everything fixed, and vary ONE impairment factor at a time:

  * rise/fall edge time        (2-15 ms)
  * receiver CW-filter width   (60/90/150/250/400 Hz)
  * dah/dit ratio              (2.6-3.4)
  * inter-element/gap scaling  (0.65-1.15)
  * MP3 encode/decode bitrate  (none / 64 / 48 / 32 / 24 / 16 kbps)

Each factor level reuses the SAME message + fist + noise realisation as the
baseline (paired comparison), so the CER delta is attributable to that factor
alone. Ranking = how far the worst realistic level pushes CER above baseline.

Run CONDITIONED for exp15 (a conditioned model):
  DEEPFIST_CONDITION=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe \
      tools/stress_test_realism.py --ckpt runs/exp15/model.pt --clips 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.morse.timing import wpm_to_timing, morph_timing
from deepfist.synth.text import random_message
from deepfist.synth.keyer import text_to_segments, segments_to_envelope
from deepfist.synth.fist import apply_fist, FistParams
from deepfist.synth.tone import envelope_to_audio
from deepfist.synth.channel import add_awgn, apply_rx_filter
from deepfist.synth.codec import mp3_roundtrip, ffmpeg_available
from deepfist.morse.alphabet import text_to_tokens, tokens_to_text
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.train.metrics import cer

SR = SAMPLE_RATE
WINDOW_S = 6.0

# Baseline (held fixed while one factor is swept).
BASE = dict(rise_ms=5.0, rx_width_hz=250.0, dahdit=3.0, gap_scale=1.0,
            mp3_kbps=None, snr_db=6.0)

# Factor sweeps: realistic ranges from HANDOFF §17.6.
SWEEPS = {
    "rise_ms":      [2.0, 5.0, 8.0, 12.0, 15.0],
    "rx_width_hz":  [60.0, 90.0, 150.0, 250.0, 400.0],
    "dahdit":       [2.6, 2.8, 3.0, 3.2, 3.4],
    "gap_scale":    [0.65, 0.8, 1.0, 1.15],
    "mp3_kbps":     [None, 64, 48, 32, 24, 16],
}


def _fit_text(rng, wpm, max_frac=0.85):
    """A message whose keyed duration fits `max_frac` of the window at `wpm`."""
    timing = wpm_to_timing(wpm)
    toks = text_to_tokens(random_message(rng, max_tokens=12))
    while toks:
        cand = tokens_to_text(toks).strip()
        if cand:
            dur = sum(s.duration for s in text_to_segments(cand, timing))
            if dur <= WINDOW_S * max_frac:
                return cand
        toks = toks[:-1]
    return "E"


def build_clip(text, wpm, pitch, *, rise_ms, rx_width_hz, dahdit, gap_scale,
               mp3_kbps, snr_db, fist_seed, noise_seed):
    """Render one controlled clip with exactly the given factor settings."""
    n = int(WINDOW_S * SR)
    timing = morph_timing(wpm_to_timing(wpm), dahdit_ratio=dahdit, gap_scale=gap_scale)
    segs = text_to_segments(text, timing)
    # Modest fixed fist so the paired clips share the same human-timing realisation.
    segs = apply_fist(segs, np.random.default_rng(fist_seed),
                      FistParams(jitter_sigma=0.06, weight=0.0))
    env = segments_to_envelope(segs, SR, rise=rise_ms / 1000.0)
    audio = envelope_to_audio(env, SR, pitch)
    if len(audio) > n:
        audio = audio[:n]
    clip = np.zeros(n, dtype=np.float32)
    clip[:len(audio)] = audio
    # Same noise realisation across factor levels (paired).
    clip = add_awgn(clip, snr_db, np.random.default_rng(noise_seed))
    clip = apply_rx_filter(clip, SR, pitch, rx_width_hz)
    if mp3_kbps:
        clip = mp3_roundtrip(clip, SR, int(mp3_kbps))
    return np.clip(clip, -1.0, 1.0).astype(np.float32)


def load_model(ckpt: Path, device: str):
    cfg_path = ckpt.parent / "config.json"
    width, ds = 1.0, 2
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        width, ds = c.get("width", 1.0), c.get("time_downsample", 2)
    net = CwCtcNet(time_downsample=ds, width=width).to(device)
    net.load_state_dict(torch.load(str(ckpt), map_location=device))
    net.eval()
    return net


@torch.no_grad()
def decode(net, audio, device):
    spec = audio_to_spectrogram(maybe_condition(audio, SR), SR)
    lp = net(spec.unsqueeze(0).unsqueeze(0).to(device))
    return greedy_ctc_decode(lp)[0]


def mean_cer(net, clips, device):
    tot = 0.0
    for audio, label in clips:
        pred = decode(net, audio, device)
        tot += cer(pred.replace(" ", ""), label.replace(" ", ""))
    return tot / len(clips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/exp15/model.pt")
    ap.add_argument("--clips", type=int, default=40, help="clips per (factor-level, WPM)")
    ap.add_argument("--wpm", type=float, nargs="+", default=[30.0, 35.0, 40.0])
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    net = load_model(ckpt, device)
    if not ffmpeg_available():
        print("WARNING: ffmpeg not found -> MP3 sweep will be a no-op")

    # Pre-draw a fixed pool of (text, pitch, fist_seed, noise_seed) per (clip, wpm),
    # shared across every factor level so comparisons are paired.
    rng = np.random.default_rng(args.seed)
    pool = {}
    for w in args.wpm:
        items = []
        for i in range(args.clips):
            text = _fit_text(rng, w)
            pitch = float(rng.uniform(560, 640))       # near conditioner OUT_PITCH
            items.append((text, pitch, int(rng.integers(1 << 30)), int(rng.integers(1 << 30))))
        pool[w] = items

    def clips_for(overrides):
        params = dict(BASE)
        params.update(overrides)
        out = []
        for w in args.wpm:
            for (text, pitch, fs, ns) in pool[w]:
                audio = build_clip(text, w, pitch, fist_seed=fs, noise_seed=ns, **params)
                out.append((audio, text))
        return out

    print(f"ckpt={ckpt}  device={device}  clips/level={args.clips}  wpm={args.wpm}  "
          f"conditioned={'DEEPFIST_CONDITION' in __import__('os').environ}")
    t0 = time.time()
    baseline = mean_cer(net, clips_for({}), device)
    print(f"\nBASELINE (clean single-signal @ {BASE['snr_db']:.0f}dB): "
          f"CER {baseline*100:.1f}%\n")

    results = {"baseline": baseline, "factors": {}}
    ranking = []
    for factor, levels in SWEEPS.items():
        print(f"[{factor}]")
        rows = []
        worst = baseline
        for lv in levels:
            c = mean_cer(net, clips_for({factor: lv}), device)
            rows.append((lv, c))
            worst = max(worst, c)
            tag = "  <-- baseline" if (BASE.get(factor) == lv) else ""
            print(f"   {str(lv):>6} : CER {c*100:5.1f}%{tag}")
        results["factors"][factor] = {"levels": [[l, c] for l, c in rows],
                                      "worst": worst, "delta": worst - baseline}
        ranking.append((factor, worst - baseline, worst))
        print()

    ranking.sort(key=lambda r: r[1], reverse=True)
    print("=== RANKING (CER increase vs baseline; higher = matters more) ===")
    print(f"baseline CER = {baseline*100:.1f}%   (real off-air target ~5%)")
    for factor, delta, worst in ranking:
        print(f"   {factor:12s}  +{delta*100:4.1f} pts  (worst {worst*100:.1f}%)")
    results["ranking"] = [{"factor": f, "delta": d, "worst": w} for f, d, w in ranking]

    print(f"\ntotal {time.time()-t0:.0f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
