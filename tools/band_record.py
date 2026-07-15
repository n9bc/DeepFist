"""Continuous wide-filter band recorder — the high-yield harvest front end.

Instead of tuning station-by-station, widen the CW filter so a whole slice of the
contest segment lands in the RX audio at once, and record it continuously. Later,
tools/band_extract.py aligns the RBN spot log (from rbn_spot_logger.py, which must
run concurrently) against this recording and cuts one labeled clip per spotted
station. One 10-15 min sitting -> dozens of labeled clips (HANDOFF §18.22 reverse
pipeline). The operator just parks the dial at the low edge of the activity; we do
not retune during the record.

We never move the dial except to place it (optional --dial); we DO widen the RX
filter for the duration and restore it on exit.

  # 1) start the spot logger (separate process):
  #    .venv/Scripts/python.exe tools/rbn_spot_logger.py --call N9BC
  # 2) park Lyra at the low edge of the CW activity (e.g. 14025), then:
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/band_record.py \
      --minutes 12 --width 3000 --sideband CWU --out runs/band
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tci_decode import parse_packet, TYPE_RX_AUDIO
from tools.rbn_harvest import parse_vfo_hz


def filter_band(sideband: str, guard: float, width: float) -> tuple[int, int]:
    """Audio passband (lo, hi) Hz covering `width` Hz of spectrum starting `guard` Hz
    off the dial. CWU puts the covered RF above the dial (positive audio); CWL below."""
    sb = sideband.upper()
    if sb.endswith("L") or sb == "LSB":
        return int(-(guard + width)), int(-guard)
    return int(guard), int(guard + width)


async def record(args):
    lo, hi = filter_band(args.sideband, args.guard, args.width)
    sr_seen = {"sr": None, "ch": None}
    chunks: list[np.ndarray] = []
    orig_filter = None  # TCI doesn't reliably report filter; we restore to a sane CW default

    async with websockets.connect(args.uri, max_size=None, ping_interval=None) as ws:
        ready = False
        dial = {"hz": None}

        async def reader():
            nonlocal ready
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    for c in msg.split(";"):
                        if c.lower().startswith("ready"):
                            ready = True
                    hz = parse_vfo_hz(msg, args.rx)
                    if hz is not None:
                        dial["hz"] = hz
                    continue
                pkt = parse_packet(msg)
                if not pkt:
                    continue
                dtype, prx, sr, ch, samples = pkt
                if dtype != TYPE_RX_AUDIO or prx != args.rx or samples.size == 0:
                    continue
                if sr_seen["sr"] is None:
                    sr_seen["sr"], sr_seen["ch"] = sr, ch
                mono = samples.reshape(-1, ch).mean(axis=1) if ch > 1 else samples
                chunks.append(mono.astype(np.float32))

        task = asyncio.create_task(reader())
        t0 = time.time()
        while not ready and time.time() - t0 < 5:
            await asyncio.sleep(0.1)

        if args.dial:
            await ws.send(f"VFO:{args.rx},0,{int(args.dial)};")
        await ws.send(f"MODULATION:{args.rx},cw;")
        await ws.send(f"RX_FILTER_BAND:{args.rx},{lo},{hi};")
        await asyncio.sleep(0.4)
        if dial["hz"] is None:
            dial["hz"] = args.dial or 0

        start_utc = datetime.now(timezone.utc)
        await ws.send("AUDIO_STREAM_SAMPLE_TYPE:float32;")
        await ws.send(f"AUDIO_START:{args.rx};")
        print(f"[band] dial {dial['hz']} Hz  filter {lo}..{hi} Hz ({args.sideband}, {args.width:.0f}Hz wide)  "
              f"recording {args.minutes:.0f} min...", flush=True)
        rec_start = time.time()
        last_report = rec_start
        while time.time() - rec_start < args.minutes * 60:
            await asyncio.sleep(0.5)
            if time.time() - last_report >= 30:
                secs = time.time() - rec_start
                got = sum(len(c) for c in chunks)
                print(f"[band] {secs/60:.1f} min  {got/(sr_seen['sr'] or 48000):.0f}s audio", flush=True)
                last_report = time.time()
        stop_utc = datetime.now(timezone.utc)
        await ws.send(f"AUDIO_STOP:{args.rx};")
        task.cancel()
        # restore a normal narrow CW filter so we don't leave the operator wide open
        try:
            await ws.send(f"RX_FILTER_BAND:{args.rx},{int(args.pitch-250)},{int(args.pitch+250)};")
        except Exception:  # noqa: BLE001 -- best effort on exit
            pass

    sr = sr_seen["sr"] or 48000
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    out = Path(args.out) / f"{start_utc:%Y-%m-%d_%H%M%S}_{(dial['hz'] or 0)//1000}kHz_band"
    out.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out / "band.wav"), sr, audio.astype(np.float32))
    session = {
        "dial_hz": int(dial["hz"] or 0),
        "sideband": args.sideband,
        "filter_lo_hz": lo, "filter_hi_hz": hi, "width_hz": args.width, "guard_hz": args.guard,
        "pitch_hz": args.pitch,
        "sr": sr,
        "t_start": start_utc.timestamp(), "t_stop": stop_utc.timestamp(),
        "start_utc": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stop_utc": stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(len(audio) / sr, 1),
    }
    (out / "session.json").write_text(json.dumps(session, indent=2))
    print(f"[band] wrote {out/'band.wav'}  ({len(audio)/sr:.0f}s, peak {np.abs(audio).max():.1f})", flush=True)
    print(f"[band] next: tools/band_extract.py --rec {out} --spots runs/rbn_spots_live.jsonl", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--uri", default="ws://127.0.0.1:40001")
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--width", type=float, default=5000.0,
                    help="spectrum width Hz to capture in the audio. Lyra caps the CW "
                         "filter at ~6 kHz (measured), so keep <=5000 or clips beyond the "
                         "real passband come out empty")
    ap.add_argument("--guard", type=float, default=200.0, help="Hz off the dial where the passband starts")
    ap.add_argument("--sideband", default="CWU")
    ap.add_argument("--pitch", type=float, default=650.0, help="pitch to restore the narrow filter to on exit")
    ap.add_argument("--dial", type=int, default=0, help="optional: set the dial (Hz) before recording (0=leave)")
    ap.add_argument("--out", default="runs/band")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(record(args)))


if __name__ == "__main__":
    main()
