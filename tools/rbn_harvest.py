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

    def __init__(self, window_s=120.0, min_skimmers=4, freq_tol_khz=0.3, cooldown_s=600.0):
        self.window_s = window_s
        self.min_skimmers = min_skimmers
        self.freq_tol = freq_tol_khz
        self.cooldown_s = cooldown_s
        self._spots: list[Spot] = []
        self._cooldown: dict[str, float] = {}

    def add(self, spot: Spot) -> None:
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
                 audio, our_pick: str, margin: float, created: "datetime | None" = None) -> Path:
    """Persist a confirmed clip as a Lyra-shaped recording dir (session.json + audio.wav)
    so tools/rbn_confirm.py and eval_real_session.py can consume it unchanged."""
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
        "decode": {"our_pick": our_pick, "margin": margin, "agree": True},
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


async def tune_once(uri, rx, dial_hz, sideband):
    """Open a short-lived TCI WS, set CW mode + dial, give the radio a moment, close."""
    import websockets
    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        await ws.send(f"MODULATION:{rx},cw;")
        await ws.send(f"VFO:{rx},0,{int(dial_hz)};")
        await asyncio.sleep(0.4)


async def _capture(uri, dwell, rx):
    """Capture `dwell` seconds of RX audio -> (sr, audio). Reuses scripts/tci_capture."""
    from scripts.tci_capture import capture
    return await capture(uri, dwell, rx)


def _is_active(audio, sr):
    """True if keyed CW is present (spot not faded). Reuses scripts/tci_decode."""
    import numpy as np
    from scripts.tci_decode import cw_activity
    active, _pitch = cw_activity(np.asarray(audio, dtype=np.float32), sr)
    return bool(active)


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
    await tune_once(args.uri, args.rx, dial, args.sideband)
    sr, audio = await _capture(args.uri, args.dwell, args.rx)

    base = {"call": cand.call, "freq_khz": cand.freq_khz, "utc": _utc_now(),
            "skimmers": cand.skimmers, "snr": cand.peak_snr, "wpm": cand.wpm}

    if not _is_active(audio, sr):
        rec = {**base, "our_pick": None, "margin": None, "agree": False, "note": "faded"}
        append_manifest(args.out, rec)
        buf._cooldown[cand.call] = now + 60.0   # short retry lockout on a faded spot
        print(f"[miss ] {cand.call} faded (no keyed CW)", flush=True)
        return rec

    picks = _decode_picks(sr, audio, args.ckpt)
    agree = cand.call in picks
    our_pick = cand.call if agree else (max(picks, key=picks.get) if picks else None)
    margin = picks.get(our_pick) if our_pick else None
    rec = {**base, "our_pick": our_pick, "margin": margin, "agree": agree}
    append_manifest(args.out, rec)
    if agree:
        d = write_sample(args.out, cand, args.sideband, dial, sr, audio, cand.call, margin)
        print(f"[AGREE] {cand.call} <-> ours {margin:.1f} nats  saved {d.name}", flush=True)
    else:
        print(f"[miss ] RBN {cand.call} != ours {our_pick}", flush=True)
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
                     cooldown_s=args.cooldown)
    stop = asyncio.Event()
    orig_vfo = await read_vfo(args.uri, args.rx)
    if orig_vfo is not None:
        print(f"[tci] original VFO {orig_vfo} Hz (restored on exit)", flush=True)
    else:
        print("[tci] VFO unknown at startup -- dial will NOT be restored on exit", flush=True)
    feed = asyncio.create_task(telnet_feed(args.call, buf, stop))
    print(f"harvesting: >={args.min_skimmers} skimmers, ranges={ranges or 'ALL'}, "
          f"dwell {args.dwell}s -> {args.out}", flush=True)
    try:
        while True:
            if await _chase_once(buf, args, time.time(), ranges) is None:
                await asyncio.sleep(2.0)
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
    ap.add_argument("--spot-window", type=float, default=120.0, dest="spot_window")
    ap.add_argument("--cooldown", type=float, default=600.0)
    ap.add_argument("--dwell", type=float, default=18.0)
    ap.add_argument("--pitch", type=float, default=650.0)
    ap.add_argument("--sideband", default="CWU")
    ap.add_argument("--ckpt", default="runs/exp15/model.pt")
    ap.add_argument("--out", default="runs/rbn_harvest")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
