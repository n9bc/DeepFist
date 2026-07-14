"""Validate capture timing: is each clip's RBN-labeled callsign actually IN the audio,
or did the reactive trigger drift off the spotted CQ? Two modes:

  (default) matcher-only  — no archive needed. For each clip, run the model-independent
      fingerprint matcher (locate_callsign_template) for its labeled call. High correlation
      => the call is verifiably present. Gives a LOWER BOUND on "call present" and a cleaner
      accuracy denominator than the raw 8/100.

  --archive  — the full reverse/RBN check (needs runs/rbn_cache/<YYYYMMDD>.csv for the clip
      day; does NOT auto-download). For each clip, gather EVERY RBN `dx` call within
      +/-window s and +/-freq-tol kHz of the recording, matcher-verify each, and report the
      spot-time-vs-audio drift. Use --utc-offset to convert session 'created' to UTC.

Clip sets: --clips eval|train|all  (eval = data/callsign_eval_heldout.txt).
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from scipy.io import wavfile

import locate_callsign_template as L
import rbn_confirm as RC

CACHE = ROOT / "runs" / "rbn_cache"


def clip_dirs(which: str):
    names = []
    if which in ("eval", "all"):
        f = ROOT / "data" / "callsign_eval_heldout.txt"
        if f.exists():
            names += f.read_text().split()
    if which in ("train", "all"):
        f = ROOT / "data" / "callsign_train_pool.txt"
        if f.exists():
            names += f.read_text().split()
    out = []
    for n in names:
        for d in ("rbn_anchors", "rbn_harvest"):
            sj = ROOT / "runs" / d / n / "session.json"
            if sj.exists():
                out.append(sj); break
    return out


def matcher_only(clips, thresh):
    present = 0
    rows = []
    for sj in clips:
        m = json.loads(sj.read_text())
        call = m["rbn"]["callsign"].upper()
        wpm = m["rbn"].get("wpm") or 20
        wav = sj.parent / m["audio"][0]["file"]
        if not wav.exists():
            continue
        sr, a = wavfile.read(str(wav))
        r, t0, t1 = L.locate(a, sr, call, wpm)
        ok = r >= thresh
        present += ok
        rows.append((call, r, ok))
    rows.sort(key=lambda x: -x[1])
    n = len(rows)
    print(f"=== matcher-only presence check ({n} clips, r>={thresh}) ===")
    print(f"labeled call VERIFIABLY present: {present}/{n} = {present/n:.0%}  (lower bound)")
    print(f"not confirmed (absent OR too hard for matcher): {n-present}/{n}\n")
    print("  weakest 12 (likely drifted-off or very hard):")
    for call, r, ok in rows[-12:]:
        print(f"    {call:9} r={r:.2f} {'present' if ok else 'NOT confirmed'}")
    print("\nNOTE: matcher misses some present-but-sloppy calls, so this UNDER-counts")
    print("presence. It cannot alone prove absence — the RBN-archive --archive mode (spot")
    print("times) is the clean test. Among CONFIRMED-present clips, model accuracy is the")
    print("honest callsign metric (run eval_callsign_heldout on that subset next).")


def archive_check(clips, window_s, freq_tol, utc_offset, thresh):
    from collections import Counter
    drift, present_labeled, multi = [], 0, 0
    missing_days = Counter()
    used = 0
    for sj in clips:
        m = json.loads(sj.read_text())
        call = m["rbn"]["callsign"].upper()
        wpm = m["rbn"].get("wpm") or 20
        f_khz = m["freqHz"] / 1000.0
        created = datetime.fromisoformat(m["created"]) + timedelta(hours=utc_offset)
        ymd = created.strftime("%Y%m%d")
        csv = CACHE / f"{ymd}.csv"
        if not csv.exists():
            missing_days[ymd] += 1
            continue
        used += 1
        t_lo = created - timedelta(seconds=window_s)
        t_hi = created + timedelta(seconds=window_s + 18)   # clip is ~18 s long
        agg = RC.scan(csv, f_khz - freq_tol, f_khz + freq_tol, t_lo, t_hi)
        wav = sj.parent / m["audio"][0]["file"]
        sr, a = wavfile.read(str(wav))
        found_here = []
        for dx, info in agg.items():
            r, t0, t1 = L.locate(a, sr, dx, int(np.median(info["wpm"])) if info["wpm"] else wpm)
            if r >= thresh:
                found_here.append((dx, r, t0, info["first"]))
        if any(dx == call for dx, *_ in found_here):
            present_labeled += 1
        if len(found_here) > 1:
            multi += 1
        for dx, r, t0, first in found_here:
            # audio time t0 vs spot time (seconds of spot after clip start)
            drift.append((first - created).total_seconds())
    print(f"=== RBN-archive validation (+/-{window_s}s, +/-{freq_tol}kHz, utc_offset={utc_offset}h) ===")
    if missing_days:
        print("  MISSING archive days (not in runs/rbn_cache): "
              + ", ".join(f"{d}({n})" for d, n in missing_days.items()))
    if not used:
        print("  no clips had a cached archive day -> cannot run. Fetch the day CSV first.")
        return
    print(f"  clips checked: {used}")
    print(f"  labeled call matcher-present within window: {present_labeled}/{used}")
    print(f"  clips with >1 RBN call present (multi-station): {multi}/{used}")
    if drift:
        d = np.array(drift)
        print(f"  spot-time minus clip-start (s): median={np.median(d):.1f} "
              f"p10={np.percentile(d,10):.1f} p90={np.percentile(d,90):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="eval", choices=["eval", "train", "all"])
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--freq-tol", type=float, default=0.3, dest="freq_tol")
    ap.add_argument("--utc-offset", type=float, default=0.0, dest="utc_offset")
    ap.add_argument("--thresh", type=float, default=0.6)
    args = ap.parse_args()
    clips = clip_dirs(args.clips)
    print(f"clip set '{args.clips}': {len(clips)} recordings\n")
    if args.archive:
        archive_check(clips, args.window, args.freq_tol, args.utc_offset, args.thresh)
    else:
        matcher_only(clips, args.thresh)


if __name__ == "__main__":
    main()
