"""Forced-align a long ARRL/W1AW code-practice recording to its KNOWN transcript, slicing
into real_all_train-style (audio, text) training windows. This is how the 4pm W1AW capture
becomes clean real training data (extends runs/real_all_train, which exp16 trains on).

Method (robust for clean practice audio, where exp16 decodes ~96%): slide non-overlapping
windows, greedy-decode each, and align the decode to the known transcript by a forward
edit-distance search to recover that window's true text span. Output: <out>/labels.jsonl +
6 s clips, ready for `train.py --wmr <out>`.
"""
from __future__ import annotations
import json
import re
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
from segment_anchors import _ed

WIN = 6.0


def clean_text(raw: str) -> str:
    """Uppercase, collapse whitespace, keep only tokenizable chars (letters/digits/=/?)."""
    s = re.sub(r"[^A-Z0-9=?/, ]", " ", raw.upper())
    return re.sub(r"\s+", " ", s).strip()


def align_window(greedy: str, full: str, pos: int) -> tuple[str, int]:
    """Find the span of `full` (searching forward from pos) that best matches the window's
    greedy decode; return (label, new_pos). Empty greedy -> advance by a guess."""
    g = greedy.strip()
    if not g:
        return "", min(len(full), pos + 6)
    lo = max(0, pos - 4)
    hi = min(len(full), pos + len(g) + 20)
    region = full[lo:hi]
    best = (10**9, "", pos)
    for start in range(len(region)):
        for wlen in range(max(1, len(g) - 6), len(g) + 7):
            end = start + wlen
            if end > len(region):
                break
            cand = region[start:end]
            d = _ed(cand, g)
            if d < best[0]:
                best = (d, cand.strip(), lo + end)
    return best[1], best[2]


def align_recording(wav_path, txt_path, net, out_dir, tag):
    full = clean_text(Path(txt_path).read_text(errors="ignore"))
    sr, a = wavfile.read(str(wav_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
    dur = len(a) / sr
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pos, t, k = [], 0, 0.0, 0
    with torch.no_grad():
        while t + WIN <= dur:
            seg = a[int(t * sr):int((t + WIN) * sr)]
            seg8 = resample_poly(seg, SAMPLE_RATE, sr).astype(np.float32) if sr != SAMPLE_RATE else seg
            lp = net(audio_to_spectrogram(maybe_condition(seg8, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            g = greedy_ctc_decode(lp)[0]
            label, pos = align_window(g, full, pos)
            if label:
                fn = f"{tag}_{k:04d}.wav"
                # store int16 at SAMPLE_RATE so the WMR loader (/32768) consumes it
                peak = float(np.abs(seg8).max()) or 1.0
                wavfile.write(str(out_dir / fn), SAMPLE_RATE, (seg8 / peak * 20000).astype(np.int16))
                rows.append({"file": str((out_dir / fn).resolve()), "text": label, "greedy": g})
                k += 1
            t += WIN
    return rows


def main():
    import rescore as R
    net = R.load_net(Path("runs/exp16/model.pt"))
    wav, txt = sys.argv[1], sys.argv[2]
    out = ROOT / "data" / (sys.argv[3] if len(sys.argv) > 3 else "w1aw_aligned")
    rows = align_recording(wav, txt, net, out, Path(wav).stem)
    (out / "labels.jsonl").write_text("\n".join(json.dumps({k: r[k] for k in ("file", "text")}) for r in rows) + "\n")
    # report alignment quality: greedy-vs-label CER (low => good clean audio; label is truth)
    from deepfist.train.metrics import cer
    q = np.mean([cer(r["greedy"], r["text"]) for r in rows]) if rows else 1.0
    print(f"{Path(wav).name}: {len(rows)} windows -> {out}")
    for r in rows[:6]:
        print(f"   greedy '{r['greedy'][:34]:34}' -> label '{r['text'][:34]}'")
    print(f"mean greedy-vs-aligned-label CER: {q:.3f}  (low = clean audio, trustworthy labels)")


if __name__ == "__main__":
    main()
