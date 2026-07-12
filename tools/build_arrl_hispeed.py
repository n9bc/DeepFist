"""Orchestrate the high-speed ARRL real training/eval sets (teacher-labeled).

Builds runs/real_arrl_train + runs/real_arrl_eval from ARRL 20-40 WPM mono WAVs,
holding out the last `--holdout` seconds of each file for eval (no leakage).
Uses tools/build_real_dataset.py's teacher labeling per window.
"""
import json, shutil, subprocess, sys
from pathlib import Path
from scipy.io import wavfile

PY = sys.executable
SPEEDS = [20, 25, 30, 35, 40]
HOLDOUT = 75.0
WIN, HOP, SKIP, MINCH = 6, 3, 30, 3

root = Path("runs/real/arrl")
train = Path("runs/real_arrl_train"); ev = Path("runs/real_arrl_eval")
for d in (train, ev):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)

def dur(wav):
    sr, a = wavfile.read(str(wav)); return len(a) / sr

def build(wav, out, skip, end):
    tmp = Path(f"runs/_tmp_build")
    if tmp.exists(): shutil.rmtree(tmp)
    cmd = [PY, "tools/build_real_dataset.py", "--wav", str(wav), "--out", str(tmp),
           "--win", str(WIN), "--hop", str(HOP), "--skip-start", str(skip), "--min-chars", str(MINCH)]
    if end: cmd += ["--end", str(end)]
    subprocess.run(cmd, check=True, capture_output=True)
    return tmp

def merge(tmp, dest):
    rows = [json.loads(l) for l in (dest/"labels.jsonl").read_text().splitlines()] if (dest/"labels.jsonl").exists() else []
    n = len(rows)
    for i, r in enumerate([json.loads(l) for l in (tmp/"labels.jsonl").read_text().splitlines()]):
        fn = f"clip_{n+i:05d}.wav"; shutil.copy(tmp/r["file"], dest/fn); r["file"] = fn; rows.append(r)
    (dest/"labels.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return len(rows)

for wpm in SPEEDS:
    wav = root / f"arrl_{wpm}wpm_mono.wav"
    if not wav.exists(): print(f"skip {wpm} (missing)"); continue
    d = dur(wav)
    t_end = max(SKIP + WIN, d - HOLDOUT)
    tr = build(wav, train, SKIP, t_end); ntr = merge(tr, train)
    evb = build(wav, ev, d - HOLDOUT + 5, 0); nev = merge(evb, ev)
    print(f"{wpm}wpm dur={d:.0f}s -> train+={len(list((tr).glob('clip_*.wav')))} eval+={len(list((evb).glob('clip_*.wav')))}")

shutil.rmtree(Path("runs/_tmp_build"), ignore_errors=True)
print("TRAIN total:", len((train/'labels.jsonl').read_text().splitlines()))
print("EVAL  total:", len((ev/'labels.jsonl').read_text().splitlines()))
