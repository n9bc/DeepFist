"""Confirm decoded callsigns against the Reverse Beacon Network (independent truth).

Our real-audio eval labels are DeepCW teacher output — a CTC decoder with the same
class of failure modes as DeepFist, so "eval CER" really measures agreement with the
teacher, not truth. The RBN is a different, mature decoder fleet: dozens of CW Skimmer
instances worldwide continuously post spots (callsign, exact freq, UTC time, SNR, WPM,
spotter). If several skimmers independently spotted call C at our recording's frequency
and time, that's *independent* ground truth — it breaks the teacher ceiling for callsign
copy specifically.

This tool cross-references a Lyra recording (its session.json gives dial freq, sideband,
UTC start, duration) against the RBN daily archive and reports which real callsigns were
on the air in that slice of band+time, ranked by how many skimmers agree.

Data source (public, dev-side only — nothing ships):
  https://data.reversebeacon.net/rbn_history/YYYYMMDD.zip   (one CSV per UTC day)
  columns: callsign,de_pfx,de_cont,freq,band,dx,dx_pfx,dx_cont,mode,db,date,speed,tx_mode
           ^spotter                 ^kHz       ^spotted call        ^SNR ^UTC   ^WPM
The archive publishes only AFTER a UTC day closes, so same-day clips must wait ~1 day.

Matching:
  * frequency — RBN `freq` is the signal's true carrier; our session logs the DIAL freq.
    For CWU the signal sits at dial + audio_pitch (≈0.3–1.5 kHz); for CWL, dial − pitch.
    So we match `freq ∈ [dial−lo, dial+hi]` with a sideband-aware, tunable margin.
  * time — spots within [start−pad, start+dur+pad] UTC (skimmer/clock slop, default 90 s).
  * mode — tx_mode == CW.
Candidates (`dx`) are ranked by distinct-skimmer count, then peak SNR.

Usage:
  # confirm one or more Lyra recording folders (auto-fetches the right RBN day)
  .venv/Scripts/python.exe tools/rbn_confirm.py --rec "C:/Users/.../2026-07-12_025923_14009kHz_CWU"
  # explicit slice (dial kHz, sideband, UTC ISO, seconds)
  .venv/Scripts/python.exe tools/rbn_confirm.py --freq 7010 --sideband CWU --utc 2026-07-11T00:02:40 --dur 20
  # self-test the matcher against a known 07-11 multi-skimmer cluster
  .venv/Scripts/python.exe tools/rbn_confirm.py --selftest

Ports to diddle's Rust host later: a background "spot confirmer" next to ScpDb::correct
could pull the RBN telnet feed live and green-light decoded calls the moment a skimmer
agrees. This offline tool nails down the matching logic first.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "runs" / "rbn_cache"
ARCHIVE_URL = "https://data.reversebeacon.net/rbn_history/{ymd}.zip"
UA = "Mozilla/5.0 (DeepFist rbn_confirm)"

# Sideband-aware default frequency margins (kHz) relative to the DIAL frequency.
# CWU signal = dial + pitch, CWL signal = dial − pitch; small opposite slop covers
# zero-beat / rounding. Widen with --margin if a signal is far up the passband.
MARGIN_HI = 2.2   # how far ABOVE dial to look (USB pitch)
MARGIN_LO = 2.2   # how far BELOW dial to look (LSB pitch)
TIME_PAD_S = 90   # UTC slop each side of the clip


def archive_csv(ymd: str) -> Path | None:
    """Return a local path to the extracted RBN CSV for YYYYMMDD, fetching+caching
    it if needed. None if the day isn't published yet (archive 404)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE / f"{ymd}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
    url = ARCHIVE_URL.format(ymd=ymd)
    print(f"  fetching {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
    except Exception as e:  # 404 = not published yet, or network error
        print(f"  archive for {ymd} unavailable ({e}). RBN publishes a day only after "
              f"it closes (00:00Z next day) -- re-run once it's up.")
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
        if not name:
            print(f"  no CSV inside {ymd}.zip"); return None
        csv_path.write_bytes(z.read(name))
    print(f"  cached {csv_path} ({csv_path.stat().st_size // (1 << 20)} MB)")
    return csv_path


def freq_window(dial_khz: float, sideband: str, margin: float | None):
    hi = margin if margin is not None else MARGIN_HI
    lo = margin if margin is not None else MARGIN_LO
    sb = sideband.upper()
    if sb.endswith("U") or sb == "USB":
        return dial_khz - 0.3, dial_khz + hi
    if sb.endswith("L") or sb == "LSB":
        return dial_khz - lo, dial_khz + 0.3
    return dial_khz - lo, dial_khz + hi  # unknown → symmetric


def scan(csv_path: Path, f_lo: float, f_hi: float, t_lo: datetime, t_hi: datetime):
    """Stream the day CSV, keep CW spots inside the freq+time window, aggregate by
    spotted call (`dx`). Returns {dx: {spotters:set, spots:int, snr:[..], wpm:[..],
    freqs:[..], first:dt, last:dt}}."""
    agg: dict[str, dict] = defaultdict(
        lambda: {"spotters": set(), "spots": 0, "snr": [], "wpm": [], "freqs": [],
                 "first": None, "last": None})
    with csv_path.open(newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if row.get("tx_mode") != "CW":
                continue
            try:
                f = float(row["freq"])
            except (ValueError, KeyError):
                continue
            if not (f_lo <= f <= f_hi):
                continue
            try:
                ts = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError):
                continue
            if not (t_lo <= ts <= t_hi):
                continue
            dx = row["dx"].strip().upper()
            if not dx:
                continue
            a = agg[dx]
            a["spotters"].add(row["callsign"])
            a["spots"] += 1
            try:
                a["snr"].append(int(row["db"]))
            except (ValueError, KeyError):
                pass
            try:
                a["wpm"].append(int(row["speed"]))
            except (ValueError, KeyError):
                pass
            a["freqs"].append(f)
            a["first"] = ts if a["first"] is None else min(a["first"], ts)
            a["last"] = ts if a["last"] is None else max(a["last"], ts)
    return agg


def report(agg: dict, top: int = 8):
    if not agg:
        print("  no RBN spots in this freq+time window.")
        return []
    ranked = sorted(
        agg.items(),
        key=lambda kv: (len(kv[1]["spotters"]), max(kv[1]["snr"] or [0])),
        reverse=True,
    )
    print(f"  {'call':10} {'skimmers':>8} {'spots':>6} {'peakSNR':>7} {'wpm':>4}  "
          f"{'freq(kHz)':>10}  {'UTC span':>17}")
    for dx, a in ranked[:top]:
        med_f = sorted(a["freqs"])[len(a["freqs"]) // 2]
        snr = max(a["snr"]) if a["snr"] else 0
        wpm = sorted(a["wpm"])[len(a["wpm"]) // 2] if a["wpm"] else 0
        span = f"{a['first'].strftime('%H:%M:%S')}-{a['last'].strftime('%H:%M:%S')}"
        star = "  <-- multi-skimmer" if len(a["spotters"]) >= 2 else ""
        print(f"  {dx:10} {len(a['spotters']):8d} {a['spots']:6d} {snr:6d}  {wpm:4d}  "
              f"{med_f:10.1f}  {span:>17}{star}")
    return [dx for dx, _ in ranked]


def confirm_slice(dial_khz, sideband, utc_start: datetime, dur_s: float,
                  margin=None, pad=TIME_PAD_S, expect=None):
    f_lo, f_hi = freq_window(dial_khz, sideband, margin)
    t_lo = utc_start - timedelta(seconds=pad)
    t_hi = utc_start + timedelta(seconds=dur_s + pad)
    print(f"slice: dial {dial_khz} kHz {sideband}  freq [{f_lo:.1f},{f_hi:.1f}]  "
          f"UTC [{t_lo:%Y-%m-%d %H:%M:%S}, {t_hi:%H:%M:%S}]")
    # a clip may straddle UTC midnight → scan both days
    days = {t_lo.strftime("%Y%m%d"), t_hi.strftime("%Y%m%d")}
    merged: dict = {}
    for ymd in sorted(days):
        cp = archive_csv(ymd)
        if cp is None:
            continue
        part = scan(cp, f_lo, f_hi, t_lo, t_hi)
        for dx, a in part.items():
            if dx not in merged:
                merged[dx] = a
            else:
                m = merged[dx]
                m["spotters"] |= a["spotters"]; m["spots"] += a["spots"]
                m["snr"] += a["snr"]; m["wpm"] += a["wpm"]; m["freqs"] += a["freqs"]
                m["first"] = min(m["first"], a["first"]); m["last"] = max(m["last"], a["last"])
    ranked = report(merged)
    if expect:
        e = expect.strip().upper()
        if e in merged:
            print(f"  [OK] CONFIRMED {e}: {len(merged[e]['spotters'])} skimmers "
                  f"(rank {ranked.index(e)+1}/{len(ranked)})")
        else:
            print(f"  [--] {e} not spotted in this window")
    return merged


def confirm_recording(rec_dir: Path, margin=None, pad=TIME_PAD_S):
    sj = json.loads((rec_dir / "session.json").read_text())
    dial = sj["freqHz"] / 1000.0
    mode = sj.get("mode", "CWU")
    start = datetime.strptime(sj["created"], "%Y-%m-%dT%H:%M:%S")  # UTC per user
    dur = float(sj.get("durationSec", 0))
    print(f"\n=== {rec_dir.name}  ({mode} {dial:.1f} kHz, {dur:.0f}s @ {start:%Y-%m-%d %H:%M:%S}Z) ===")
    return confirm_slice(dial, mode, start, dur, margin=margin, pad=pad)


def selftest():
    """Validate the matcher on real 07-11 data: K1GHL was spotted by 5 skimmers near
    7010 kHz at 00:02:4x. A synthetic 'recording' there should recover K1GHL."""
    print("SELF-TEST against cached 20260711 (K1GHL cluster @ ~7010 kHz):")
    m = confirm_slice(7010.0, "CWU", datetime(2026, 7, 11, 0, 2, 30), 30,
                      margin=2.5, expect="K1GHL")
    ok = "K1GHL" in m and len(m["K1GHL"]["spotters"]) >= 2
    print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rec", action="append", default=[], help="Lyra recording dir (repeatable)")
    ap.add_argument("--freq", type=float, help="dial frequency kHz (explicit slice)")
    ap.add_argument("--sideband", default="CWU", help="CWU/CWL/USB/LSB")
    ap.add_argument("--utc", help="clip start, ISO e.g. 2026-07-12T02:29:10 (UTC)")
    ap.add_argument("--dur", type=float, default=20.0, help="clip duration s")
    ap.add_argument("--margin", type=float, default=None, help="freq match margin kHz (override)")
    ap.add_argument("--pad", type=float, default=TIME_PAD_S, help="UTC time pad s each side")
    ap.add_argument("--expect", help="assert this call appears (exit non-zero if not)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.rec:
        for d in args.rec:
            confirm_recording(Path(d), margin=args.margin, pad=args.pad)
        return
    if args.freq and args.utc:
        confirm_slice(args.freq, args.sideband,
                      datetime.strptime(args.utc, "%Y-%m-%dT%H:%M:%S"),
                      args.dur, margin=args.margin, pad=args.pad, expect=args.expect)
        return
    ap.error("give --rec DIR, or --freq + --utc, or --selftest")


if __name__ == "__main__":
    main()
