"""Passive RBN spot logger — feeds the operator-driven capture flow.

Unlike rbn_harvest.py, this NEVER touches the radio. It just streams the RBN
telnet feed and appends every CW spot (with a wall-clock timestamp) to a JSONL
file. tools/capture_here.py then reads that file to label whatever frequency the
operator has manually tuned to. This is the "record first, RBN as timestamped
label" pipeline (HANDOFF §18.22) with the operator choosing the signal.

  .venv/Scripts/python.exe tools/rbn_spot_logger.py --call N9BC --out runs/rbn_spots_live.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rbn_harvest import parse_rbn_line  # reuse the vetted RBN line parser


async def run(call: str, out: Path, keep_s: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    backoff = 1.0
    while True:
        try:
            reader, writer = await asyncio.open_connection("telnet.reversebeacon.net", 7000)
            writer.write((call + "\r\n").encode())
            await writer.drain()
            backoff = 1.0
            print(f"[rbn] connected as {call}; logging CW spots -> {out}", flush=True)
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                sp = parse_rbn_line(raw.decode("ascii", "ignore"))
                if not sp:
                    continue
                rec = {"t": sp.t, "call": sp.call, "freq_khz": sp.freq_khz,
                       "spotter": sp.spotter, "snr": sp.snr_db, "wpm": sp.wpm}
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001 -- reconnect on any socket fault
            print(f"[rbn] {e}; reconnecting in {backoff:.0f}s", flush=True)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", required=True, help="callsign for the RBN telnet login")
    ap.add_argument("--out", default="runs/rbn_spots_live.jsonl")
    ap.add_argument("--keep-s", type=float, default=900.0, help="(reserved) rolling retention")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.call, Path(args.out), args.keep_s))
    except KeyboardInterrupt:
        print("\n[rbn] stopped.", flush=True)


if __name__ == "__main__":
    main()
