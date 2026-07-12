"""CTC hypothesis rescorer — WSJT-X/JT65 "Deep Search" adapted to CTC.

WSJT-X Deep Search doesn't try to *read* a weak message; it asks "of all the
messages this station could plausibly be sending, which best matches the
received symbols?" This tool does the same for CW: instead of snapping greedy
TEXT to MASTER.SCP by edit distance (tools/scp_correct.py), it scores candidate
callsigns directly against the model's CTC log-probability lattice —

    nll = F.ctc_loss(log_probs, candidate_ids, blank=0, reduction="sum")

— and picks the likeliest. The lattice can separate candidates that text
distance can't: WP3Z vs WP3B vs WP3C are all 1 edit from a garbled WP3?, but
their final-character acoustics score very differently under the model.

Decode-time only: no training, no model changes; exp15 stays deployed.

Modes
  RESCORE (default) — greedy-decode the clip, find callsign-shaped words
    (diddle's liberal CALL_RE, len>=3, not a CQ/TEST/599-style marker), generate
    SCP candidates within --max-edit of each, swap each candidate into the full
    label sequence, score the full sequence, keep the argmin. Prints per word:
    original -> winner, margin to runner-up (nats), top-3.
  PROBE (--probe CALL, repeatable) — score the given candidates directly as the
    sole content of the clip (or the --t0/--t1 slice): for clips whose greedy
    output is too garbled to contain a swappable word.

Usage
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/rescore.py --wav clip.wav
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/rescore.py \
      --wav clip.wav --probe XE2TT --probe XE2T --t0 2 --t1 8

Honors DEEPFIST_CONDITION=1 (front-end conditioner) like every eval tool.

Porting note: the winning logic (candidate generation from SCP + CTC-lattice
scoring of each swap) belongs in diddle's Rust host next to ScpDb::correct —
the ONNX session already exposes the same log_probs; a Rust ctc_loss over a
handful of candidates is a few hundred multiply-adds per call.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID

# Mirrors diddle CwDecoderView.svelte — liberal call shape + non-call markers.
CALL_RE = re.compile(r"^([A-Z0-9]{1,3}/)?[A-Z]{1,2}\d{1,3}[A-Z]{1,4}(/[A-Z0-9]{1,3})?$")
MARKERS = {
    "CQ", "DE", "TEST", "QRZ", "TU", "73", "88", "599", "5NN", "5N",
    "K", "KN", "SK", "AGN", "PSE", "NR", "R",
}


def is_call(tok: str) -> bool:
    c = tok.upper()
    return len(c) >= 3 and CALL_RE.match(c) is not None and c not in MARKERS


def load_scp(path: Path) -> list[str]:
    calls = set()
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#") and not s.startswith("!"):
            calls.add(s)
    return sorted(calls)


def edit_distance_capped(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, early-exit to cap+1 once it can't stay <= cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def scp_candidates(word: str, scp: list[str], max_edit: int, max_cands: int) -> list[str]:
    """SCP calls within max_edit of word, nearest first, capped at max_cands."""
    scored = []
    for c in scp:
        d = edit_distance_capped(word, c, max_edit)
        if d <= max_edit:
            scored.append((d, c))
    scored.sort()
    return [c for _, c in scored[:max_cands]]


def load_audio(path: Path, t0: float | None, t1: float | None) -> np.ndarray:
    sr, a = wavfile.read(str(path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
    lo = int((t0 or 0.0) * sr)
    hi = int(t1 * sr) if t1 is not None else len(a)
    a = a[lo:hi]
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a


def load_net(ckpt: Path) -> CwCtcNet:
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    net = CwCtcNet(time_downsample=cfg["time_downsample"], width=cfg["width"])
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


def sequence_nll(log_probs: torch.Tensor, text: str) -> float:
    """Full-sequence CTC negative log-likelihood of `text` under the lattice.

    inf for untokenizable text or a target longer than the lattice (CTC needs
    target_len <= T; with repeated symbols it needs even more frames, which
    ctc_loss itself reports as inf)."""
    try:
        tokens = text_to_tokens(text)
    except ValueError:
        return math.inf
    ids = [TOKEN_TO_ID[t] for t in tokens]
    T = log_probs.shape[0]
    if not ids or len(ids) > T:
        return math.inf
    target = torch.tensor(ids, dtype=torch.long).unsqueeze(0)          # [1,L]
    nll = F.ctc_loss(log_probs, target,
                     input_lengths=torch.tensor([T]),
                     target_lengths=torch.tensor([len(ids)]),
                     blank=0, reduction="sum", zero_infinity=False)
    v = float(nll)
    return v if math.isfinite(v) else math.inf


def fmt_top(ranked: list[tuple[float, str]], k: int = 3) -> str:
    return "  ".join(f"{c} {nll:.1f}" for nll, c in ranked[:k])


def rescore(log_probs: torch.Tensor, greedy: str, scp: list[str],
            max_edit: int, max_cands: int) -> None:
    words = greedy.split()
    call_idxs = [i for i, w in enumerate(words) if is_call(w)]
    if not call_idxs:
        print("no callsign-shaped words in greedy output — try --probe mode")
        return
    for i in call_idxs:
        word = words[i]
        cands = scp_candidates(word, scp, max_edit, max_cands)
        if word not in cands:
            cands.insert(0, word)   # original always competes
        ranked = []
        for c in cands:
            trial = words[:i] + [c] + words[i + 1:]
            ranked.append((sequence_nll(log_probs, " ".join(trial)), c))
        ranked.sort()
        best_nll, best = ranked[0]
        margin = (ranked[1][0] - best_nll) if len(ranked) > 1 else math.inf
        flag = "" if best == word else "  <-- RESCORED"
        print(f"  {word:10} -> {best:10} margin {margin:6.2f} nats "
              f"({len(cands)} cands)   top3: {fmt_top(ranked)}{flag}")


def probe(log_probs: torch.Tensor, candidates: list[str]) -> None:
    ranked = sorted((sequence_nll(log_probs, c.upper()), c.upper()) for c in candidates)
    best_nll = ranked[0][0]
    print(f"  {'candidate':10} {'nll':>8}  {'vs best':>8}")
    for nll, c in ranked:
        print(f"  {c:10} {nll:8.1f}  {nll - best_nll:+8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="runs/exp15/model.pt")
    ap.add_argument("--scp", default="data/MASTER.SCP")
    ap.add_argument("--max-edit", type=int, default=2)
    ap.add_argument("--max-cands", type=int, default=60)
    ap.add_argument("--probe", action="append", default=[],
                    help="score CALL as the sole clip content (repeatable)")
    ap.add_argument("--t0", type=float, default=None, help="slice start (s)")
    ap.add_argument("--t1", type=float, default=None, help="slice end (s)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    a = load_audio(Path(args.wav), args.t0, args.t1)
    a = maybe_condition(a, SAMPLE_RATE)
    net = load_net(root / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt))
    with torch.no_grad():
        log_probs = net(audio_to_spectrogram(a, SAMPLE_RATE).unsqueeze(0).unsqueeze(0))  # [T,1,C]

    greedy = greedy_ctc_decode(log_probs)[0]
    span = f" [{args.t0 or 0:.1f}s..{args.t1 if args.t1 is not None else len(a)/SAMPLE_RATE:.1f}s]" \
        if (args.t0 is not None or args.t1 is not None) else ""
    print(f"{Path(args.wav).name}{span}  T={log_probs.shape[0]} frames")
    print(f"greedy: {greedy!r}")
    print(f"greedy nll: {sequence_nll(log_probs, greedy):.1f}")

    if args.probe:
        probe(log_probs, args.probe)
    else:
        scp = load_scp(root / args.scp if not Path(args.scp).is_absolute() else Path(args.scp))
        rescore(log_probs, greedy, scp, args.max_edit, args.max_cands)


if __name__ == "__main__":
    main()
