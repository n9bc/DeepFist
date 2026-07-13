"""Standalone scheduled capture of the W1AW code-practice session (real off-air, labeled
later from ARRL's posted text). Tunes 20m code-practice freq in CW and records N seconds.

Designed to be run by Windows Task Scheduler at 2100 UTC (4pm CDT) independent of any agent
session. Stops the band collector first (frees the radio), captures, restores nothing (the
operator retunes). Output: runs/w1aw/<UTC>_14047kHz.wav
"""
from __future__ import annotations
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from scipy.io import wavfile

URI = "ws://127.0.0.1:40001"
DIAL_HZ = 14047500          # 14.0475 MHz — W1AW 20m CW code practice
SECONDS = 1800              # 30 minutes (cover a full code-practice session)


async def run():
    import websockets
    from scripts.tci_capture import capture
    # tune to the code-practice freq in CW (freq set only; assumes CW mode)
    try:
        async with websockets.connect(URI, max_size=None, ping_interval=None) as ws:
            await ws.send("MODULATION:0,cw;")
            await ws.send(f"VFO:0,0,{DIAL_HZ};")
            await asyncio.sleep(1.0)
    except Exception as e:
        print(f"[tune] failed: {e} — is Lyra/TCI up on {URI}?", flush=True)
        return
    print(f"capturing {SECONDS}s at {DIAL_HZ/1e6:.4f} MHz ...", flush=True)
    sr, audio = await capture(URI, SECONDS, 0)
    out = ROOT / "runs" / "w1aw"; out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}_14047kHz.wav"
    wavfile.write(str(fn), sr, np.asarray(audio, np.float32))
    peak = float(np.abs(audio).max()); rms = float(np.sqrt((audio ** 2).mean()))
    print(f"wrote {fn}  ({len(audio)/sr:.0f}s, peak={peak:.1f} rms={rms:.2f})", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
