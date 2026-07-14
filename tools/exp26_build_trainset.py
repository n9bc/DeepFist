"""Build the exp26 training extension: turn the leakage-safe callsign_train_pool anchors
into clean (6 s window, callsign) training clips, using segment_anchors' locate+verify.

ONLY processes data/callsign_train_pool.txt (callsign-disjoint from the 100-clip eval), so
nothing from the eval leaks in. Writes data/realset_train_pool/ (does NOT touch the existing
data/realset_train). Run with DEEPFIST_CONDITION=1.

Then emits data/real_ext26/labels.jsonl = real_ext (2420, exp24 known-good) + the new pool
clips, ready for: scripts/train.py --wmr data/real_ext26 (exp16 recipe).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from scipy.io import wavfile

import segment_anchors as S
from deepfist.features.spectrogram import SAMPLE_RATE

OUT = ROOT / "data" / "realset_train_pool"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net = S.load_net()
    names = (ROOT / "data" / "callsign_train_pool.txt").read_text().split()
    rows, kept = [], 0
    for n in names:
        sj = None
        for d in ("rbn_anchors", "rbn_harvest"):
            p = ROOT / "runs" / d / n / "session.json"
            if p.exists():
                sj = p; break
        if sj is None:
            continue
        m = json.loads(sj.read_text())
        call = m["rbn"]["callsign"].upper()
        wav = sj.parent / m["audio"][0]["file"]
        if not wav.exists():
            continue
        clip = S.segment_anchor(wav, call, net)
        status = "no-locate"
        if clip is not None and len(clip) >= int(0.4 * SAMPLE_RATE):
            clip = clip[:S.WINDOW]
            padded = np.zeros(S.WINDOW, dtype=np.float32)
            padded[:len(clip)] = clip
            peak = float(np.abs(padded).max()) or 1.0
            clip16 = (padded / peak * 20000.0).astype(np.int16)
            fn = f"{call}_{kept:03d}.wav"
            wavfile.write(str(OUT / fn), SAMPLE_RATE, clip16)
            rows.append({"file": str((OUT / fn).resolve()), "text": call})
            kept += 1
            status = f"kept {round(len(clip)/SAMPLE_RATE,2)}s"
        print(f"  {call:9} -> {status}", flush=True)
    (OUT / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nclean pool pairs: {kept}/{len(names)} -> {OUT}")

    # emit exp26 training list = real_ext (known-good) + new pool clips
    base = [l for l in (ROOT / "data" / "real_ext" / "labels.jsonl").read_text().splitlines() if l.strip()]
    ext = [json.dumps(r) for r in rows]
    out = ROOT / "data" / "real_ext26"
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.jsonl").write_text("\n".join(base + ext) + "\n")
    print(f"wrote data/real_ext26/labels.jsonl = {len(base)} (real_ext) + {len(ext)} (pool) = {len(base)+len(ext)}")


if __name__ == "__main__":
    main()
