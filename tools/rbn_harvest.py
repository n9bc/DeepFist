"""Live RBN-confirmed CW sample harvester.

Watches the RBN telnet feed (telnet.reversebeacon.net:7000); when >=4 distinct
skimmers independently spot the same callsign at the same frequency, tunes the
radio there over TCI, records the real off-air audio, decodes it with DeepFist,
and -- if our decode agrees with the RBN call -- saves the clip as an independently
labeled real-audio training sample. See
docs/superpowers/specs/2026-07-12-rbn-live-harvest-design.md.

Live-only (no replay abstraction). Reuses scripts/tci_capture, scripts/tci_decode,
and tools/rbn_confirm unmodified. Run with DEEPFIST_CONDITION=1 (exp15 is conditioned):

  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/rbn_harvest.py \
      --call <YOURCALL> --range 14000-14100 --dwell 18
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for scripts.*
sys.path.insert(0, str(Path(__file__).resolve().parent))       # tools/, for rbn_confirm

_RBN_RE = re.compile(
    r"DX de\s+(?P<spotter>[A-Z0-9/#\-]+):\s+"
    r"(?P<freq>\d+\.\d+)\s+"
    r"(?P<call>[A-Z0-9/]+)\s+"
    r"(?P<mode>[A-Z0-9]+)\s+"
    r"(?P<snr>-?\d+)\s*dB\s+"
    r"(?P<wpm>\d+)\s*wpm",
    re.IGNORECASE,
)


@dataclass
class Spot:
    spotter: str
    freq_khz: float
    call: str
    snr_db: int
    wpm: int
    t: float


def parse_rbn_line(line: str, now: float | None = None) -> "Spot | None":
    """Parse one RBN telnet spot line into a Spot, or None if not a CW spot."""
    m = _RBN_RE.search(line or "")
    if not m or m.group("mode").upper() != "CW":
        return None
    spotter = m.group("spotter").upper().split("-")[0].rstrip("#")
    return Spot(
        spotter=spotter,
        freq_khz=float(m.group("freq")),
        call=m.group("call").upper(),
        snr_db=int(m.group("snr")),
        wpm=int(m.group("wpm")),
        t=now if now is not None else time.time(),
    )


def is_us_spotter(call: str) -> bool:
    """True if `call` is a US amateur callsign (K/N/W prefix, or A followed by A-L).
    Used to keep only spots from skimmers that actually hear the signal in the US, so we
    tune to stations reaching the operator's antenna rather than DX only distant sites hear."""
    c = (call or "").upper().split("/")[0]
    if not c:
        return False
    if c[0] in ("K", "N", "W"):
        return True
    return c[0] == "A" and len(c) >= 2 and "A" <= c[1] <= "L"


def is_keyed_signal(audio, sr, cov_thresh: float = 0.65) -> bool:
    """True if the audio contains an on/off-KEYED tone (real CW), not just energy.

    Distinguishes a genuine CW station from AGC-amplified band noise in the pitch-centered
    CW filter: the envelope coefficient-of-variation (std/mean) of narrow-band noise is
    ~0.52 (Rayleigh), while keyed CW swings between key-down and key-up for ~0.8+. Measured
    live: real stations 0.76-0.86, empty-frequency AGC hiss 0.51-0.58."""
    import numpy as np
    a = np.asarray(audio, dtype=np.float64)
    if a.size < sr // 2:                       # need >= 0.5 s to judge keying
        return False
    w = np.hanning(len(a))
    S = np.abs(np.fft.rfft(a * w))
    f = np.fft.rfftfreq(len(a), 1.0 / sr)
    m = (f >= 300) & (f <= 1000)
    if not m.any() or S[m].max() <= 0:
        return False
    tone = float(f[m][np.argmax(S[m])])
    t = np.arange(len(a)) / sr
    k = max(1, int(0.004 * sr))
    kern = np.ones(k) / k
    i = np.convolve(a * np.cos(2 * np.pi * tone * t), kern, "same")
    q = np.convolve(a * np.sin(2 * np.pi * tone * t), kern, "same")
    env = np.sqrt(i * i + q * q)
    if env.mean() <= 0:
        return False
    return float(env.std() / env.mean()) > cov_thresh


# --- tuning math + band filter ---------------------------------------------------
def dial_hz_for(signal_khz: float, sideband: str, pitch_hz: float) -> int:
    """Dial frequency (Hz) that places the RBN signal at our model's audio pitch.
    Mirrors tools/rbn_confirm.freq_window: CWU signal = dial + pitch, CWL = dial - pitch."""
    signal_hz = signal_khz * 1000.0
    sb = sideband.upper()
    if sb.endswith("L") or sb == "LSB":
        return int(round(signal_hz + pitch_hz))
    return int(round(signal_hz - pitch_hz))


def parse_range(spec: str) -> tuple[float, float]:
    lo, hi = spec.split("-")
    return float(lo), float(hi)


def in_range(freq_khz: float, ranges: list[tuple[float, float]]) -> bool:
    if not ranges:
        return True
    return any(lo <= freq_khz <= hi for lo, hi in ranges)


def cw_filter_band(sideband: str, pitch_hz: float, width_hz: float) -> tuple[int, int]:
    """Audio passband (low, high) Hz for a narrow CW filter of `width_hz` centered on the
    pitch. CWU sits on the positive (USB) audio side, CWL on the negative (LSB) side."""
    half = width_hz / 2.0
    sb = sideband.upper()
    if sb.endswith("L") or sb == "LSB":
        return int(-(pitch_hz + half)), int(-(pitch_hz - half))
    return int(pitch_hz - half), int(pitch_hz + half)


def parse_vfo_hz(msg, rx: int) -> "int | None":
    """Extract the dial frequency (Hz) for `rx` from a TCI 'vfo:rx,chan,freq;' status
    string, or None if this message carries no VFO for that receiver. TCI emits a state
    burst on connect that includes the current VFO (same field scripts/tci_decode reads)."""
    if not isinstance(msg, str):
        return None
    for cmd in msg.strip().split(";"):
        if not cmd.lower().startswith("vfo:"):
            continue
        parts = cmd.split(":", 1)[1].split(",")
        try:
            if int(parts[0]) == rx:
                return int(float(parts[-1]))
        except (ValueError, IndexError):
            continue
    return None


# --- consensus gate --------------------------------------------------------------
@dataclass
class Candidate:
    call: str
    freq_khz: float
    skimmers: int
    peak_snr: int
    wpm: int
    spotters: list


class SpotBuffer:
    """Rolling aggregator that fires when >=min_skimmers distinct spotters agree on
    a call within a frequency cluster and time window. Per-call cooldown after chase."""

    def __init__(self, window_s=120.0, min_skimmers=4, freq_tol_khz=0.3, cooldown_s=600.0,
                 us_only=True):
        self.window_s = window_s
        self.min_skimmers = min_skimmers
        self.freq_tol = freq_tol_khz
        self.cooldown_s = cooldown_s
        self.us_only = us_only
        self._spots: list[Spot] = []
        self._cooldown: dict[str, float] = {}

    def add(self, spot: Spot) -> None:
        if self.us_only and not is_us_spotter(spot.spotter):
            return                             # only US skimmers count toward consensus
        self._spots.append(spot)

    def mark_chased(self, call: str, now: float) -> None:
        self._cooldown[call.upper()] = now + self.cooldown_s

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        self._spots = [s for s in self._spots if s.t >= cutoff]

    def _densest_cluster(self, spots: list[Spot]) -> list[Spot]:
        """Return the on-frequency subset with the most distinct spotters (each spot
        acts as a cluster anchor; membership = within freq_tol kHz of the anchor)."""
        best: list[Spot] = []
        for anchor in spots:
            grp = [s for s in spots if abs(s.freq_khz - anchor.freq_khz) <= self.freq_tol]
            if len({s.spotter for s in grp}) > len({s.spotter for s in best}):
                best = grp
        return best

    def ready_candidates(self, now: float, ranges=None) -> list[Candidate]:
        self._evict(now)
        by_call: dict[str, list[Spot]] = defaultdict(list)
        for s in self._spots:
            by_call[s.call].append(s)
        cands: list[Candidate] = []
        for call, spots in by_call.items():
            if self._cooldown.get(call, 0.0) > now:
                continue
            cluster = self._densest_cluster(spots)
            spotters = sorted({s.spotter for s in cluster})
            if len(spotters) < self.min_skimmers:
                continue
            freqs = sorted(s.freq_khz for s in cluster)
            med_freq = freqs[len(freqs) // 2]
            if ranges and not in_range(med_freq, ranges):
                continue
            wpms = sorted(s.wpm for s in cluster)
            cands.append(Candidate(
                call=call,
                freq_khz=med_freq,
                skimmers=len(spotters),
                peak_snr=max(s.snr_db for s in cluster),
                wpm=wpms[len(wpms) // 2],
                spotters=spotters,
            ))
        cands.sort(key=lambda c: (c.skimmers, c.peak_snr), reverse=True)
        return cands


# --- corpus sink -----------------------------------------------------------------
def write_sample(out_root, cand: Candidate, sideband: str, dial_hz: int, sr: int,
                 audio, our_pick, margin, agree: bool = True,
                 created: "datetime | None" = None) -> Path:
    """Persist a >=4-skimmer-confirmed clip as a Lyra-shaped recording dir (session.json +
    audio.wav) so tools/rbn_confirm.py and eval_real_session.py can consume it unchanged.
    The RBN consensus (rbn.callsign) is the training label; decode.* is our model's attempt
    and `agree` records whether it matched -- saved regardless so hard cases aren't lost."""
    import numpy as np
    from scipy.io import wavfile
    created = created or datetime.now(timezone.utc)
    name = f"{created:%Y-%m-%d_%H%M%S}_{cand.freq_khz:.0f}kHz_{cand.call}"
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(d / "audio.wav"), sr, np.asarray(audio, dtype=np.float32))
    session = {
        "freqHz": int(dial_hz),
        "mode": sideband,
        "created": created.strftime("%Y-%m-%dT%H:%M:%S"),
        "durationSec": round(len(audio) / sr, 2),
        "audio": [{"file": "audio.wav"}],
        "rbn": {
            "callsign": cand.call, "skimmers": cand.skimmers,
            "peak_snr": cand.peak_snr, "wpm": cand.wpm, "spotters": cand.spotters,
        },
        "decode": {"our_pick": our_pick, "margin": margin, "agree": bool(agree)},
    }
    (d / "session.json").write_text(json.dumps(session, indent=2))
    return d


def append_manifest(out_root, record: dict) -> None:
    Path(out_root).mkdir(parents=True, exist_ok=True)
    with (Path(out_root) / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- live I/O seams (module-level so they can be monkeypatched in tests) ----------
async def read_vfo(uri, rx, timeout=3.0):
    """Read the current dial frequency (Hz) for `rx` from the TCI state burst emitted on
    connect. Returns None if nothing arrives in `timeout` s (used to restore on exit)."""
    import websockets
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            end = time.time() + timeout
            while time.time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=max(0.05, end - time.time()))
                except asyncio.TimeoutError:
                    break
                hz = parse_vfo_hz(msg, rx)
                if hz is not None:
                    return hz
    except Exception as e:  # noqa: BLE001 -- best-effort; absence just disables restore
        print(f"[tci] could not read VFO: {e}", flush=True)
    return None


async def set_vfo(uri, rx, dial_hz):
    """Set only the dial frequency for `rx` (used to restore the original VFO on exit)."""
    import websockets
    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(f"VFO:{rx},0,{int(dial_hz)};")
        await asyncio.sleep(0.2)


async def tune_once(uri, rx, dial_hz, sideband, pitch=650.0, filter_hz=0.0):
    """Open a short-lived TCI WS, set CW mode + dial, optionally narrow the CW filter to
    isolate the target from adjacent QRM, give the radio a moment, close."""
    import websockets
    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(f"MODULATION:{rx},cw;")
        await ws.send(f"VFO:{rx},0,{int(dial_hz)};")
        if filter_hz and filter_hz > 0:
            lo, hi = cw_filter_band(sideband, pitch, filter_hz)
            await ws.send(f"RX_FILTER_BAND:{rx},{lo},{hi};")
        await asyncio.sleep(0.4)


async def _capture(uri, dwell, rx):
    """Capture `dwell` seconds of RX audio -> (sr, audio). Reuses scripts/tci_capture."""
    from scripts.tci_capture import capture
    return await capture(uri, dwell, rx)


def _is_active(audio, sr, cov_thresh=0.65):
    """True if on/off-KEYED CW is present -- not merely energy in the passband. Uses the
    envelope-CoV test so AGC-amplified band noise (which fills the pitch-centered CW filter
    on an empty frequency) is NOT mistaken for a station. See is_keyed_signal."""
    return is_keyed_signal(audio, sr, cov_thresh)


def _decode_picks(sr, audio, ckpt):
    """Decode+rescore -> {call: margin}. Writes a temp WAV and reuses rbn_confirm."""
    import tempfile
    import numpy as np
    from scipy.io import wavfile
    import rbn_confirm
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "clip.wav"
        wavfile.write(str(wav), sr, np.asarray(audio, dtype=np.float32))
        return rbn_confirm.decode_recording(wav, ckpt)


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# --- orchestration ---------------------------------------------------------------
async def _chase_once(buf, args, now, ranges):
    """One iteration: pick the top ready candidate, tune, capture, gate, decode,
    log to the manifest, and save a labeled sample on agreement. Returns the record
    (or None if no candidate is ready)."""
    cands = buf.ready_candidates(now, ranges)
    if not cands:
        return None
    cand = cands[0]
    buf.mark_chased(cand.call, now)
    dial = dial_hz_for(cand.freq_khz, args.sideband, args.pitch)
    print(f"[chase] {cand.call} {cand.freq_khz:.1f}kHz {cand.skimmers} skimmers "
          f"snr{cand.peak_snr} -> dial {dial} Hz", flush=True)
    await tune_once(args.uri, args.rx, dial, args.sideband, args.pitch, args.filter_hz)

    base = {"call": cand.call, "freq_khz": cand.freq_khz, "utc": _utc_now(),
            "skimmers": cand.skimmers, "snr": cand.peak_snr, "wpm": cand.wpm}

    # Verify a signal is actually present BEFORE recording: a short probe listen avoids
    # spending a full dwell (and a decode) on a spot that has already gone quiet.
    psr, paudio = await _capture(args.uri, args.probe, args.rx)
    if not _is_active(paudio, psr, args.key_cov):
        rec = {**base, "our_pick": None, "margin": None, "agree": False,
               "note": "no-signal", "saved": None}
        append_manifest(args.out, rec)
        buf._cooldown[cand.call] = now + 60.0   # short retry lockout on a dead spot
        print(f"[skip ] {cand.call} no signal at spot -- not recorded", flush=True)
        return rec

    # Signal confirmed -> record the full sample.
    sr, audio = await _capture(args.uri, args.dwell, args.rx)
    if not _is_active(audio, sr, args.key_cov):
        # Signal was there at probe but stopped mid-record (e.g. QSO ended) -> no label value.
        rec = {**base, "our_pick": None, "margin": None, "agree": False,
               "note": "faded", "saved": None}
        append_manifest(args.out, rec)
        buf._cooldown[cand.call] = now + 60.0
        print(f"[skip ] {cand.call} faded during record -- not saved", flush=True)
        return rec

    picks = _decode_picks(sr, audio, args.ckpt)
    agree = cand.call in picks
    our_pick = cand.call if agree else (max(picks, key=picks.get) if picks else None)
    margin = picks.get(our_pick) if our_pick else None
    # The >=4-skimmer RBN consensus IS the ground-truth label, independent of whether our
    # (weaker) model agreed -> always harvest the real audio; `agree` is recorded metadata.
    d = write_sample(args.out, cand, args.sideband, dial, sr, audio, our_pick, margin, agree=agree)
    rec = {**base, "our_pick": our_pick, "margin": margin, "agree": agree, "saved": d.name}
    append_manifest(args.out, rec)
    tag = "AGREE" if agree else "label"
    mtxt = f"{margin:.1f}" if margin is not None else "--"
    print(f"[{tag:5}] {cand.call} ({cand.skimmers} skim) ours={our_pick} m={mtxt}  "
          f"saved {d.name}", flush=True)
    return rec


async def telnet_feed(call, buf, stop):
    """Stream the RBN telnet feed into `buf` until `stop` is set. Reconnects w/ backoff."""
    backoff = 1.0
    while not stop.is_set():
        try:
            reader, writer = await asyncio.open_connection("telnet.reversebeacon.net", 7000)
            writer.write((call + "\r\n").encode())
            await writer.drain()
            backoff = 1.0
            while not stop.is_set():
                raw = await reader.readline()
                if not raw:
                    break
                sp = parse_rbn_line(raw.decode("ascii", "ignore"))
                if sp:
                    buf.add(sp)
        except Exception as e:  # noqa: BLE001 -- reconnect on any socket/parse fault
            print(f"[telnet] {e}; reconnecting in {backoff:.0f}s", flush=True)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


async def run(args):
    ranges = [parse_range(r) for r in args.range]
    buf = SpotBuffer(window_s=args.spot_window, min_skimmers=args.min_skimmers,
                     cooldown_s=args.cooldown, us_only=args.us_only)
    stop = asyncio.Event()
    orig_vfo = await read_vfo(args.uri, args.rx)
    if orig_vfo is not None:
        print(f"[tci] original VFO {orig_vfo} Hz (restored on exit)", flush=True)
    else:
        print("[tci] VFO unknown at startup -- dial will NOT be restored on exit", flush=True)
    feed = asyncio.create_task(telnet_feed(args.call, buf, stop))
    print(f"harvesting: >={args.min_skimmers} skimmers, ranges={ranges or 'ALL'}, "
          f"dwell {args.dwell}s -> {args.out}", flush=True)
    from websockets.exceptions import WebSocketException
    tci_down = (OSError, asyncio.TimeoutError, WebSocketException)
    try:
        while True:
            try:
                if await _chase_once(buf, args, time.time(), ranges) is None:
                    await asyncio.sleep(2.0)
            except tci_down as e:
                # Radio/TCI dropped (e.g. the SDR app restarted) -- survive it and keep the
                # RBN feed + spot buffer alive; retry shortly instead of crashing the session.
                print(f"[tci] radio unreachable ({e.__class__.__name__}); retrying in 5s",
                      flush=True)
                await asyncio.sleep(5.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nstopping.", flush=True)
    finally:
        stop.set()
        feed.cancel()
        if orig_vfo is not None:
            try:
                await set_vfo(args.uri, args.rx, orig_vfo)
                print(f"[tci] restored VFO to {orig_vfo} Hz", flush=True)
            except Exception as e:  # noqa: BLE001 -- exit path, log and move on
                print(f"[tci] VFO restore failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--call", required=True, help="your callsign (RBN telnet login)")
    ap.add_argument("--uri", default="ws://127.0.0.1:40001", help="TCI WS (Lyra default)")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--range", action="append", default=[],
                    help="receivable kHz window LO-HI (repeatable); default = accept all")
    ap.add_argument("--min-skimmers", type=int, default=4, dest="min_skimmers")
    ap.add_argument("--any-spotter", action="store_false", dest="us_only",
                    help="count spots from any skimmer (default: US spotters only)")
    ap.add_argument("--key-cov", type=float, default=0.65,
                    help="min envelope CoV to accept a signal as keyed CW (vs AGC noise)")
    ap.add_argument("--spot-window", type=float, default=120.0, dest="spot_window")
    ap.add_argument("--cooldown", type=float, default=600.0)
    ap.add_argument("--dwell", type=float, default=18.0)
    ap.add_argument("--probe", type=float, default=3.0,
                    help="short pre-capture listen (s) to confirm a signal before recording")
    ap.add_argument("--pitch", type=float, default=650.0)
    ap.add_argument("--filter-hz", type=float, default=400.0, dest="filter_hz",
                    help="narrow CW filter width Hz centered on --pitch (0 = leave radio filter)")
    ap.add_argument("--sideband", default="CWU")
    ap.add_argument("--ckpt", default="runs/exp15/model.pt")
    ap.add_argument("--out", default="runs/rbn_harvest")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
