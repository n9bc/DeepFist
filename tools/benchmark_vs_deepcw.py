"""Head-to-head CER benchmark: DeepFist checkpoints vs DeepCW on a labeled dir.

Decodes every clip in a WebMorseRunner-labeled directory (clip_N.wav +
labels.jsonl with meta.snr) with one or more DeepFist checkpoints and, optionally,
DeepCW's reference model, then reports per-clip CER aggregated overall and per-SNR
bucket as a side-by-side table (printed + saved to runs/bench_*.{md,json}).

Licensing: DeepCW is AGPL-3.0-only. We do NOT copy its code or weights into this
repo. When --deepcw is set we import its reference decoder functions from the local
install (DEEPCW_DIR) and load its model.onnx at runtime for local evaluation only.
Nothing from DeepCW ships or is committed here.

Usage:
  .venv/Scripts/python.exe tools/benchmark_vs_deepcw.py \
      --eval-dir runs/wmr_evalA \
      --ckpt runs/exp4/model_8000.pt --ckpt runs/exp5/model_8000.pt --deepcw
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.model.net import CwCtcNet, count_params
from deepfist.model.decode import greedy_ctc_decode
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.features.conditioner import maybe_condition
from deepfist.train.metrics import cer

DEEPCW_DIR = Path(r"C:\dev\deepcw-engine")


# --------------------------------------------------------------------------- data
def load_eval(eval_dir: Path):
    """Return [(wav_path, text, snr)] from a WMR-labeled dir."""
    rows = []
    for line in (eval_dir / "labels.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((eval_dir / r["file"], r["text"], float(r["meta"]["snr"])))
    return rows


def snr_bucket(snr: float) -> int:
    return int(round(snr))


def _nospace(s: str) -> str:
    return s.replace(" ", "")


def score(preds, rows):
    """Score a model's predictions.

    Primary metric is **space-normalized** CER (spaces stripped from both pred and
    target): CW word-spacing is genuinely ambiguous, and raw CER mostly measures
    match to the label's spacing convention rather than decode quality. Raw CER is
    also reported for reference.

    Returns dict: {overall (norm), overall_raw, buckets (norm per SNR), n}.
    """
    per_bucket = defaultdict(list)
    norm_all, raw_all, exact_all = [], [], []
    for (_, text, snr), p in zip(rows, preds):
        cn = cer(_nospace(p), _nospace(text))
        norm_all.append(cn)
        raw_all.append(cer(p, text))
        exact_all.append(1.0 if _nospace(p) == _nospace(text) else 0.0)  # perfect copy
        per_bucket[snr_bucket(snr)].append(cn)
    overall = sum(norm_all) / len(norm_all) if norm_all else 0.0
    overall_raw = sum(raw_all) / len(raw_all) if raw_all else 0.0
    exact = sum(exact_all) / len(exact_all) if exact_all else 0.0
    buckets = {b: sum(v) / len(v) for b, v in per_bucket.items()}
    return {"overall": overall, "overall_raw": overall_raw, "exact": exact,
            "buckets": buckets, "n": len(norm_all)}


# ---------------------------------------------------------------------- deepfist
def decode_ours(ckpt: Path, rows, device: str, downsample: int = 2):
    # Rebuild the exact architecture from the run's config.json if present.
    cfg_path = Path(ckpt).parent / "config.json"
    width, ds = 1.0, downsample
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        width, ds = c.get("width", 1.0), c.get("time_downsample", downsample)
    net = CwCtcNet(time_downsample=ds, width=width).to(device)
    net.load_state_dict(torch.load(str(ckpt), map_location=device))
    net.eval()
    preds = []
    with torch.no_grad():
        for wav, _text, _snr in rows:
            sr, a = wavfile.read(str(wav))
            a = a.astype(np.float32) / 32768.0
            if sr != SAMPLE_RATE:
                a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
            spec = audio_to_spectrogram(maybe_condition(a, SAMPLE_RATE), SAMPLE_RATE)          # [F, T]
            lp = net(spec.unsqueeze(0).unsqueeze(0).to(device))  # [T, B, C]
            preds.append(greedy_ctc_decode(lp)[0])
    return preds


# ------------------------------------------------------------------------ deepcw
def load_deepcw():
    """Import DeepCW's reference decoder + open an ORT session (local, dev-only)."""
    src = DEEPCW_DIR / "examples" / "python" / "decode_morse.py"
    spec = importlib.util.spec_from_file_location("deepcw_ref", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = mod.load_metadata(DEEPCW_DIR / "model.onnx.json")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(DEEPCW_DIR / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    return mod, meta, sess


def decode_deepcw(mod, meta, sess, rows):
    tr = int(meta["sample_rate"])
    chars = list(meta["chars"])
    blank = int(meta["blank_index"])
    in_name, out_name = meta["onnx_input_name"], meta["onnx_output_name"]
    preds = []
    for wav, _text, _snr in rows:
        srate, audio = mod.read_wav_mono(Path(wav))
        audio = mod.resample_linear(audio, srate, tr)
        spec = mod.audio_to_spectrogram(audio, meta)             # [1,1,time,freq]
        out = sess.run([out_name], {in_name: spec})
        preds.append(mod.greedy_ctc_decode(out[0], chars, blank))
    return preds


# ----------------------------------------------------------------------- latency
def measure_latency(ckpt: Path, window_s: float = 6.0, sr: int = SAMPLE_RATE,
                    iters: int = 60, threads: int = 1):
    """Mean/p95 forward-pass time for one window on CPU, single-thread — a proxy
    for the C#/ONNX-Runtime live-decode deployment. Returns dict of ms + RTF."""
    import time as _t
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        cfg_path = Path(ckpt).parent / "config.json"
        width, ds = 1.0, 2
        if cfg_path.exists():
            c = json.loads(cfg_path.read_text())
            width, ds = c.get("width", 1.0), c.get("time_downsample", 2)
        net = CwCtcNet(time_downsample=ds, width=width).eval()
        n = int(window_s * sr)
        spec = audio_to_spectrogram(np.zeros(n, np.float32), sr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            for _ in range(5):
                net(spec)                                    # warmup
            ts = []
            for _ in range(iters):
                t0 = _t.perf_counter()
                net(spec)
                ts.append((_t.perf_counter() - t0) * 1000.0)
    finally:
        torch.set_num_threads(prev)
    ts.sort()
    mean = sum(ts) / len(ts)
    p95 = ts[int(0.95 * len(ts)) - 1]
    return {"ms_mean": mean, "ms_p95": p95, "rtf": (window_s * 1000.0) / mean,
            "params": count_params(net), "width": width}


# ------------------------------------------------------------------------ report
def surpasses(model_overall, model_b, dcw_overall, dcw_b) -> bool:
    """Success = lower overall CER AND <= DeepCW at every SNR bucket >= 0 dB."""
    if model_overall >= dcw_overall:
        return False
    for b, dc in dcw_b.items():
        if b >= 0 and model_b.get(b, 1.0) > dc + 1e-9:
            return False
    return True


def render_table(results, buckets_sorted):
    """Table of space-normalized CER: overall (norm), raw for reference, per-SNR."""
    names = list(results.keys())
    head = ["model", "CER", "(raw)", "copy%"] + [f"{b:+d}dB" for b in buckets_sorted]
    widths = [max(len(h), 7) for h in head]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for name in names:
        r = results[name]
        bk = r["buckets"]
        row = ([name, f"{r['overall']*100:.1f}%", f"{r['overall_raw']*100:.1f}%",
                f"{r.get('exact', 0)*100:.1f}%"]
               + [f"{bk.get(b, float('nan'))*100:.1f}%" for b in buckets_sorted])
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    return ("\n".join(lines) + "\n\n_CER = space-normalized (primary); (raw) includes "
            "word-spacing; copy% = overs decoded 100% correct (perfect copy)._")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="repeatable; label = parent dir name")
    ap.add_argument("--deepcw", action="store_true")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--latency", action="store_true", help="also report CPU 1-thread forward latency")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_dir = Path(args.eval_dir)
    rows = load_eval(eval_dir)
    if args.limit:
        rows = rows[: args.limit]
    print(f"eval-dir={eval_dir}  clips={len(rows)}  device={device}")

    results = {}
    for ckpt in args.ckpt:
        p = Path(ckpt)
        label = p.parent.name or p.stem
        t0 = time.time()
        preds = decode_ours(p, rows, device, args.downsample)
        results[label] = score(preds, rows)
        print(f"  {label:10s} CER {results[label]['overall']*100:5.1f}% (raw {results[label]['overall_raw']*100:.1f}%, copy {results[label]['exact']*100:.0f}%)  ({time.time()-t0:.1f}s)")

    if args.deepcw:
        t0 = time.time()
        mod, meta, sess = load_deepcw()
        preds = decode_deepcw(mod, meta, sess, rows)
        results["DeepCW"] = score(preds, rows)
        print(f"  {'DeepCW':10s} CER {results['DeepCW']['overall']*100:5.1f}% (raw {results['DeepCW']['overall_raw']*100:.1f}%, copy {results['DeepCW']['exact']*100:.0f}%)  ({time.time()-t0:.1f}s)")

    buckets_sorted = sorted({b for r in results.values() for b in r["buckets"]})
    table = render_table(results, buckets_sorted)
    print("\n" + table)

    verdict = {}
    if "DeepCW" in results:
        dcw = results["DeepCW"]
        for name, r in results.items():
            if name == "DeepCW":
                continue
            win = surpasses(r["overall"], r["buckets"], dcw["overall"], dcw["buckets"])
            verdict[name] = win
            print(f"  {name}: {'SURPASSES DeepCW' if win else 'does NOT surpass DeepCW'}"
                  f"  (norm CER {r['overall']*100:.1f}% vs {dcw['overall']*100:.1f}%)")

    latency = {}
    if args.latency:
        print("\nlatency (CPU, 1 thread, per 6s window - live-decode proxy):")
        for ckpt in args.ckpt:
            lat = measure_latency(Path(ckpt))
            latency[Path(ckpt).parent.name] = lat
            print(f"  {Path(ckpt).parent.name:10s} {lat['ms_mean']:6.1f} ms  p95 {lat['ms_p95']:6.1f} ms  "
                  f"RTF {lat['rtf']:6.1f}x  ({lat['params']:,} params, width {lat['width']})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else eval_dir.parent / f"bench_{stamp}"
    payload = {
        "eval_dir": str(eval_dir), "clips": len(rows), "buckets": buckets_sorted,
        "metric": "space-normalized CER (primary); overall_raw includes word-spacing",
        "results": results,
        "verdict_vs_deepcw": verdict,
        "latency_cpu_1thread": latency,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(
        f"# Benchmark {stamp}\n\neval: `{eval_dir}` ({len(rows)} clips)\n\n{table}\n\n"
        + "".join(f"- **{k}**: {'surpasses' if w else 'does not surpass'} DeepCW\n"
                  for k, w in verdict.items())
    )
    print(f"\nsaved: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
