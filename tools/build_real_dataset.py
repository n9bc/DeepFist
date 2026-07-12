"""Build a labeled REAL-audio dataset by pseudo-labeling with DeepCW (teacher).

HANDOFF §16.4 pipeline. Slides a window over real recordings, decodes each window
with DeepCW (which reads real off-air CW that DeepFist garbles — the whole reason
this works), keeps windows whose label is non-trivial and tokenizes into DeepFist's
alphabet, peak-normalizes the audio to match the synthetic/WMR level convention
(peak ~1.0), and writes a WMR-compatible dataset (clip_*.wav + labels.jsonl). That
dataset drops straight into training via `train.py --wmr <dir>` (BlendedDataset).

Licensing: DeepCW stays external (AGPL, dev-only). Only its decoded *text* is used
as a label; text is not copyrightable and nothing from DeepCW ships or is committed.

Optional quality gates:
  --min-chars N   drop windows with fewer than N tokens (silence / fragments)
  --agree CKPT    also decode with a DeepFist ckpt; keep window only if the two
                  decoders' space-stripped outputs have CER <= --agree-cer
                  (consensus filtering — trims teacher hallucinations). For clean
                  single-signal sources you can skip this and trust the teacher.

Usage:
  .venv/Scripts/python.exe tools/build_real_dataset.py \
     --wav runs/real/w1aw/w1aw_13wpm_mono.wav --out runs/real_w1aw \
     --win 12 --hop 8 --skip-start 40 --min-chars 6
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID
from deepfist.train.metrics import cer

_bench_src = Path(__file__).resolve().parent / "benchmark_vs_deepcw.py"
_spec = ilu.spec_from_file_location("bench", _bench_src)
_bench = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

STORE_SR = 8000  # stored clip SR (loader resamples to 3200); keeps files small


def tokenizes(text: str) -> bool:
    try:
        ids = [TOKEN_TO_ID[t] for t in text_to_tokens(text)]
        return len(ids) > 0
    except KeyError:
        return False


def load_mono(path: Path):
    sr, a = wavfile.read(str(path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float32)
    return sr, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", required=True, help="repeatable input WAV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=12.0, help="window seconds")
    ap.add_argument("--hop", type=float, default=8.0, help="hop seconds")
    ap.add_argument("--skip-start", type=float, default=0.0, help="skip leading seconds (preamble)")
    ap.add_argument("--end", type=float, default=0.0, help="stop at this second per file (0=to end); use to carve train/eval splits without leakage")
    ap.add_argument("--min-chars", type=int, default=5)
    ap.add_argument("--rms-gate", type=float, default=0.003, help="skip near-silent windows below this RMS (post-norm scale)")
    ap.add_argument("--agree", default=None, help="DeepFist ckpt for consensus filtering (optional)")
    ap.add_argument("--agree-cer", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="cap kept clips (0=all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mod, meta, sess = _bench.load_deepcw()
    tr = int(meta["sample_rate"]); chars = list(meta["chars"]); blank = int(meta["blank_index"])
    inn, outn = meta["onnx_input_name"], meta["onnx_output_name"]

    def teacher(seg, sr):
        a2 = mod.resample_linear(seg.astype(np.float32), sr, tr)
        return mod.greedy_ctc_decode(sess.run([outn], {inn: mod.audio_to_spectrogram(a2, meta)})[0], chars, blank).strip()

    student = None
    if args.agree:
        import torch
        from deepfist.model.net import CwCtcNet
        from deepfist.model.decode import greedy_ctc_decode
        from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
        cfg = Path(args.agree).parent / "config.json"; w, ds = 1.0, 2
        if cfg.exists():
            c = json.loads(cfg.read_text()); w, ds = c.get("width", 1.0), c.get("time_downsample", 2)
        net = CwCtcNet(time_downsample=ds, width=w); net.load_state_dict(torch.load(args.agree, map_location="cpu")); net.eval()

        def student(seg, sr):
            a2 = resample_poly(seg, SAMPLE_RATE, sr).astype(np.float32) if sr != SAMPLE_RATE else seg
            with torch.no_grad():
                lp = net(audio_to_spectrogram(a2, SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            return greedy_ctc_decode(lp)[0].strip()

    rows = []
    kept = skipped_silence = skipped_short = skipped_tok = skipped_agree = 0
    idx = 0
    for wav in args.wav:
        wav = Path(wav)
        sr, audio = load_mono(wav)
        dur = len(audio) / sr
        if args.end and args.end < dur:
            dur = args.end
        t = args.skip_start
        while t + args.win <= dur:
            seg = audio[int(t * sr):int((t + args.win) * sr)]
            peak = np.abs(seg).max() + 1e-9
            seg_n = (seg / peak * 0.95).astype(np.float32)
            if np.sqrt((seg_n * seg_n).mean()) < args.rms_gate:
                skipped_silence += 1; t += args.hop; continue
            label = teacher(seg_n, sr)
            toks = text_to_tokens(label) if label else []
            if len([c for c in label if c != " "]) < args.min_chars:
                skipped_short += 1; t += args.hop; continue
            if not tokenizes(label):
                skipped_tok += 1; t += args.hop; continue
            if student is not None:
                s = student(seg_n, sr)
                if cer(s.replace(" ", ""), label.replace(" ", "")) > args.agree_cer:
                    skipped_agree += 1; t += args.hop; continue
            # store downsampled, peak-normalized int16 clip
            store = resample_poly(seg_n, STORE_SR, sr).astype(np.float32)
            store = store / (np.abs(store).max() + 1e-9) * 0.95
            fn = f"clip_{idx:05d}.wav"
            wavfile.write(str(out / fn), STORE_SR, (store * 32767).astype(np.int16))
            rows.append({"file": fn, "text": label, "meta": {"snr": 99, "src": wav.name, "t0": round(t, 1)}})
            kept += 1; idx += 1
            t += args.hop
            if args.limit and kept >= args.limit:
                break
        if args.limit and kept >= args.limit:
            break

    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"kept={kept}  skipped: silence={skipped_silence} short={skipped_short} "
          f"untokenizable={skipped_tok} disagree={skipped_agree}")
    print(f"wrote {out/'labels.jsonl'}")
    for r in rows[:8]:
        print(f"  {r['file']} [{r['meta']['src']} @{r['meta']['t0']}s]  {r['text']!r}")


if __name__ == "__main__":
    main()
