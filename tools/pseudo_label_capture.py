"""Pseudo-label a long real CW capture (no known transcript) into real_all_train-style
training windows, using exp16 as the teacher. This is how a W1AW bulletin capture (plain
text, no published code-practice transcript) becomes real training data.

Same window format as align_arrl.py (6 s, int16 @ SAMPLE_RATE=3200, peak->20000) but the
label is exp16's own greedy decode (self-training) — used for the 3PM/4PM W1AW captures
where no clean transcript exists. Non-empty windows only.

Usage:
  .venv/Scripts/python.exe tools/pseudo_label_capture.py <wav> <out_tag>
  -> data/<out_tag>/labels.jsonl + 6 s clips, ready for train.py --wmr data/<out_tag>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.model.decode import greedy_ctc_decode

WIN = 6.0


def label_recording(wav_path, net, out_dir, tag):
    sr, a = wavfile.read(str(wav_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
    dur = len(a) / sr
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t, k = [], 0.0, 0
    with torch.no_grad():
        while t + WIN <= dur:
            seg = a[int(t * sr):int((t + WIN) * sr)]
            seg8 = resample_poly(seg, SAMPLE_RATE, sr).astype(np.float32) if sr != SAMPLE_RATE else seg
            lp = net(audio_to_spectrogram(maybe_condition(seg8, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            g = greedy_ctc_decode(lp)[0].strip()
            if g:
                fn = f"{tag}_{k:04d}.wav"
                peak = float(np.abs(seg8).max()) or 1.0
                wavfile.write(str(out_dir / fn), SAMPLE_RATE, (seg8 / peak * 20000).astype(np.int16))
                rows.append({"file": str((out_dir / fn).resolve()), "text": g})
                k += 1
            t += WIN
    return rows


def main():
    import rescore as R
    net = R.load_net(Path("runs/exp16/model.pt"))
    wav = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else Path(wav).stem
    out = ROOT / "data" / tag
    rows = label_recording(wav, net, out, "w1aw")
    (out / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{Path(wav).name}: {len(rows)} windows -> {out}")
    for r in rows[:6]:
        print(f"   {r['text'][:50]!r}")


if __name__ == "__main__":
    main()
