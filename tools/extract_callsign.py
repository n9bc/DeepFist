"""Contest-grammar callsign extractor — decode-time, no retraining.

Real contest callsign copy: the rescorer surfaces only ~20% because it picks the highest-
margin callsign-shaped decode across a whole QSO, often a mis-decoded exchange word. But
the call is REPEATED and follows DE/CQ ("CQ CQ DE KK7D", "DE KK7D KK7D"). Scoring
well-formed callsign tokens by repetition + a DE/CQ-precedence bonus + MASTER.SCP membership
lifts copy to ~29% on 147 real anchors. (Fuzzy SCP-snapping was tried and HURT — it
over-corrects malformed tokens onto wrong calls. Keep well-formed tokens only.)
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.decode import greedy_ctc_decode

CALL = re.compile(r"^[A-Z]{1,2}[0-9][A-Z]{1,4}$")


def extract_callsign(wav_path, net, scp_set, win=6.0, hop=1.5, blank_pen=1.0) -> "str | None":
    """Return the most likely callsign in a QSO clip via contest-grammar voting, or None.
    blank_pen counters CTC blank over-prediction on real audio (recovers dropped chars ->
    more well-formed call tokens); 1.0 is best on real anchors (32% vs 30% at pen=0)."""
    sr, a = wavfile.read(str(wav_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
    dur = len(a) / sr
    score: Counter = Counter()
    t = 0.0
    with torch.no_grad():
        while t + win <= dur + hop:
            seg = a[int(t * sr):int((t + win) * sr)]
            if len(seg) >= sr:
                x = maybe_condition(resample_poly(seg, SAMPLE_RATE, sr).astype(np.float32), SAMPLE_RATE)
                lp = net(audio_to_spectrogram(x, SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
                if blank_pen:
                    lp = lp.clone(); lp[..., 0] -= blank_pen      # BLANK_ID = 0
                toks = greedy_ctc_decode(lp)[0].split()
                for i, tk in enumerate(toks):
                    if not CALL.match(tk):
                        continue
                    s = 1
                    if i > 0 and toks[i - 1] in ("DE", "CQ"):
                        s += 3                          # call follows DE/CQ
                    if scp_set and tk in scp_set:
                        s += 1                          # a known real call
                    score[tk] += s
            t += hop
    return score.most_common(1)[0][0] if score else None


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "."); sys.path.insert(0, "tools")
    import rescore as R
    net = R.load_net(Path("runs/exp16/model.pt"))
    scp = set(R.load_scp(Path("data/MASTER.SCP")))
    anc = sorted(Path("runs/rbn_anchors").glob("*/session.json"))
    ok = 0
    for sj in anc:
        m = json.loads(sj.read_text())
        ok += (extract_callsign(sj.parent / m["audio"][0]["file"], net, scp) == m["rbn"]["callsign"].upper())
    print(f"contest-grammar callsign copy: {ok}/{len(anc)} = {ok/len(anc):.0%}")
