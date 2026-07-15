"""Teacher-label Brent's real Lyra station recordings into a training/eval set.

Lyra saves each capture as Documents/Lyra/Recordings/<stamp>_<freq>_<mode>/
audio_001.wav (+ session.json). These are the true deployment distribution — real
off-air CW off his own receiver. We DeepCW-teacher-label each (like the ARRL set),
holding out a few whole recordings for a leakage-free eval.
"""
import json, shutil, subprocess, sys
from pathlib import Path
from scipy.io import wavfile

PY = sys.executable
REC = Path(r"C:\Users\bcrie\Documents\Lyra\Recordings")
MIN_DUR = 8.0          # skip clips shorter than this
HOLDOUT = {"2026-07-12_183839_14052kHz_CWU", "2026-07-12_183910_14051kHz_CWU"}  # eval
WIN, HOP, SKIP, MINCH = 6, 3, 1, 3

train = Path("runs/real_lyra_train"); ev = Path("runs/real_lyra_eval")
for d in (train, ev):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)

def dur(w):
    sr, a = wavfile.read(str(w)); return len(a) / sr

def build(wav, out_tmp, skip):
    if out_tmp.exists(): shutil.rmtree(out_tmp)
    subprocess.run([PY, "tools/build_real_dataset.py", "--wav", str(wav), "--out", str(out_tmp),
                    "--win", str(WIN), "--hop", str(HOP), "--skip-start", str(skip),
                    "--min-chars", str(MINCH)], check=True, capture_output=True)
    return out_tmp

def merge(tmp, dest):
    rows = [json.loads(l) for l in (dest/"labels.jsonl").read_text().splitlines()] if (dest/"labels.jsonl").exists() else []
    n = len(rows)
    for i, r in enumerate([json.loads(l) for l in (tmp/"labels.jsonl").read_text().splitlines()]):
        fn = f"clip_{n+i:05d}.wav"; shutil.copy(tmp/r["file"], dest/fn); r["file"] = fn; rows.append(r)
    (dest/"labels.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return len(rows)

tmp = Path("runs/_lyra_tmp")
n_tr = n_ev = 0
for rec in sorted(p for p in REC.iterdir() if p.is_dir()):
    wav = rec / "audio_001.wav"
    if not wav.exists() or dur(wav) < MIN_DUR:
        continue
    dest = ev if rec.name in HOLDOUT else train
    b = build(wav, tmp, SKIP)
    tot = merge(b, dest)
    kept = len(list(b.glob("clip_*.wav")))
    tag = "EVAL" if dest is ev else "train"
    print(f"{rec.name}: {dur(wav):.0f}s -> {kept} clips [{tag}]")
shutil.rmtree(tmp, ignore_errors=True)
print("TRAIN total:", len((train/'labels.jsonl').read_text().splitlines()))
print("EVAL  total:", len((ev/'labels.jsonl').read_text().splitlines()) if (ev/'labels.jsonl').exists() else 0)
