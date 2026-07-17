"""BT real-capture gate: decode the operator's REAL BT recording and check the
model reads the = / == prosigns (promoted from scratch_gate3_realbt.py).

The reference is runs/bt_ref/cw_sample4.wav (48 kHz, N9BC sending textbook BTs
at ~25 wpm, incl. a double ==). A champion MUST keep reading them — exp28/exp30
proved Farnsworth training silently erodes this (HANDOFF §18.30), so this is a
standing adoption gate alongside real ARRL CER and the held-out copy eval.

  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/bt_real_eval.py \
      [runs/<exp>/model.pt ...]        (default: exp16 exp27_bt)

PASS looks like exp27_bt: '...TU 73 N9BC == 0123456789 ... = 9B8C...'
FAIL looks like exp16/exp28: BTs come out as X/B garbage or stray 5s.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPFIST_CONDITION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

from eval_real_session import decode_ours, load_wav

WAV = ROOT / "runs" / "bt_ref" / "cw_sample4.wav"


def main():
    ckpts = sys.argv[1:] or ["runs/exp16/model.pt", "runs/exp27_bt/model.pt"]
    sr, a = load_wav(str(WAV), False)
    dur = len(a) / sr
    win = max(dur + 1.0, 6.0)
    print(f"{WAV.name}  sr={sr}  dur={dur:.1f}s  peak={abs(a).max():.3f}")
    for ckpt in ckpts:
        txt = decode_ours(ckpt, a, sr, win=win, hop=win)
        n_bt = txt.count("=")
        print(f"  {Path(ckpt).parent.name:16s} ={n_bt:2d}  -> {txt!r}")


if __name__ == "__main__":
    main()
