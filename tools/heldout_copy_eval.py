"""RAW-COPY eval on the 100-clip leakage-safe real held-out (off-air, RBN labels).

Decodes each clip with raw greedy (no rescorer) and reports whether the labeled
callsign appears in the copy, split by matcher-confirmed presence (a miss on a
present-confirmed clip is a real read failure). This is the metric that showed
raw copy holds the call 14/18 (78%) while the rescorer only picked 6/18 — the
rescorer, not the acoustics, was the callsign bottleneck (HANDOFF §18.25).

  .venv/Scripts/python.exe tools/heldout_copy_eval.py [runs/<exp>/model.pt]

Baseline (exp16): 20/100 overall, 14/18 present-only.
"""
import os, sys, json
os.environ["DEEPFIST_CONDITION"] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch, json as J
import locate_callsign_template as L
from deepfist.features.spectrogram import SAMPLE_RATE, audio_to_spectrogram
from deepfist.features.conditioner import condition
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
cfg = J.load(open(str(Path(CKPT).parent / "config.json")))
net = CwCtcNet(time_downsample=cfg["time_downsample"], width=cfg["width"])
net.load_state_dict(torch.load(CKPT, map_location="cpu")); net.eval()

def clips():
    names = (ROOT/"data"/"callsign_eval_heldout.txt").read_text().split()
    for n in names:
        for d in ("rbn_anchors","rbn_harvest"):
            sj = ROOT/"runs"/d/n/"session.json"
            if sj.exists():
                m = J.loads(sj.read_text())
                wav = sj.parent/(m["audio"][0]["file"] if m.get("audio") else "audio.wav")
                yield wav, m["rbn"]["callsign"].upper(), (m["rbn"].get("wpm") or 0)
                break

def decode(a3):
    win=int(6.0*SAMPLE_RATE); hop=int(3.0*SAMPLE_RATE); parts=[]
    for s in range(0,max(1,len(a3)-win+1),hop):
        seg=a3[s:s+win]
        if len(seg)<win: seg=np.pad(seg,(0,win-len(seg)))
        with torch.no_grad():
            parts.append(greedy_ctc_decode(net(audio_to_spectrogram(condition(seg,SAMPLE_RATE),SAMPLE_RATE).unsqueeze(0).unsqueeze(0)))[0])
    return " ".join(parts)

rows=[]
for wav, call, wpm in clips():
    if not wav.exists(): continue
    sr,a=wavfile.read(str(wav)); a=a.astype(np.float32); a=a.mean(1) if a.ndim>1 else a
    r,_,_=L.locate(a,sr,call,wpm or 20)
    g=gcd(sr,SAMPLE_RATE); a3=resample_poly(a,SAMPLE_RATE//g,sr//g).astype(np.float32)
    txt=decode(a3)
    present = r>=0.6
    hit = call in txt.replace(" ","") or call in txt
    rows.append((call,wpm,r,present,hit,len(txt.replace(' ','')),txt[:50]))

n=len(rows)
present=[x for x in rows if x[3]]
print(f"real held-out: {n} clips  | matcher-confirmed present: {len(present)}")
print(f"callsign substring-hit overall: {sum(x[4] for x in rows)}/{n}")
print(f"callsign substring-hit on present-only: {sum(x[4] for x in present)}/{len(present)}")
print(f"\nwpm distribution (all): min={min(x[1] for x in rows)} med={int(np.median([x[1] for x in rows]))} max={max(x[1] for x in rows)}")
# present-only hit binned by wpm
for lo,hi in [(0,20),(20,27),(27,99)]:
    b=[x for x in present if lo<=x[1]<hi]
    if b: print(f"  present clips wpm[{lo},{hi}): hit {sum(x[4] for x in b)}/{len(b)}")
# over-insertion signature: chars produced per clip (real clips are ~18s, short calls)
print(f"\nchars decoded per clip: med={int(np.median([x[5] for x in rows]))} (short callsign clips; high => over-emitting junk)")
print("\nsample present-but-MISSED clips (call, wpm, r, decoded):")
for call,wpm,r,pres,hit,ln,txt in [x for x in present if not x[4]][:10]:
    print(f"  {call:9} wpm={wpm} r={r:.2f} | {txt!r}")
