"""Cut labeled single-signal clips out of a wide band recording (band_record.py)
using the RBN spot log as timestamped frequency labels (HANDOFF §18.22).

For each RBN spot whose time falls inside the recording and whose frequency falls
inside the recorded passband, we:
  1. heterodyne that station's audio offset down to a standard pitch (~600 Hz), so
     the target lands where the conditioner expects and neighbours shift away;
  2. bandpass around the pitch to kill the neighbours -> a clean SINGLE-signal clip;
  3. cut a window around the spot time, teacher-label it with DeepCW, and (optionally)
     keep it only if the decode contains the RBN callsign (proves we isolated the
     right station). Output is a WMR dataset (clip_*.wav + labels.jsonl) ready for
     scripts/train.py --wmr.

  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/band_extract.py \
      --rec runs/band/<stamp>_band --spots runs/rbn_spots_live.jsonl \
      --out runs/real_band_train --require-agree
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, butter, sosfiltfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID
from deepfist.train.metrics import cer

_b = Path(__file__).resolve().parent / "benchmark_vs_deepcw.py"
_sp = ilu.spec_from_file_location("bench", _b); _B = ilu.module_from_spec(_sp); _sp.loader.exec_module(_B)

STORE_SR = 8000
TARGET_PITCH = 600.0


def isolate(band_audio: np.ndarray, sr: int, audio_tone_hz: float, i0: int, i1: int,
            pitch: float = TARGET_PITCH, bw: float = 130.0) -> np.ndarray:
    """Pure DSP: pull the station at `audio_tone_hz` out of the wide recording as a
    clean single-tone clip centred on `pitch`. Heterodyne (analytic-signal shift so
    the image is suppressed) then bandpass. Returns real audio [i0:i1] at `sr`."""
    seg = band_audio[max(0, i0):min(len(band_audio), i1)].astype(np.float64)
    if seg.size < sr // 4:
        return np.zeros(0, np.float32)
    t = np.arange(seg.size) / sr
    analytic = hilbert(seg)
    shifted = np.real(analytic * np.exp(-1j * 2 * np.pi * (audio_tone_hz - pitch) * t))
    sos = butter(4, [(pitch - bw) / (sr / 2), (pitch + bw) / (sr / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, shifted).astype(np.float32)


def load_spots(path: Path, t0: float, t1: float):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t0 <= s["t"] <= t1:
            rows.append(s)
    return rows


def tokenizes(text: str) -> bool:
    try:
        return len([TOKEN_TO_ID[t] for t in text_to_tokens(text)]) > 0
    except KeyError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rec", required=True, help="a band_record.py output dir (band.wav + session.json)")
    ap.add_argument("--spots", default="runs/rbn_spots_live.jsonl")
    ap.add_argument("--out", required=True, help="WMR dataset dir to write")
    ap.add_argument("--pre", type=float, default=5.0, help="seconds before spot time")
    ap.add_argument("--post", type=float, default=8.0, help="seconds after spot time")
    ap.add_argument("--merge-s", type=float, default=20.0, help="min gap between kept clips of the same call")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--require-agree", action="store_true",
                    help="keep a clip only if the DeepCW decode contains the RBN callsign")
    ap.add_argument("--min-skimmers", type=int, default=3, help="drop spots weaker than this many skimmers")
    args = ap.parse_args()

    rec = Path(args.rec)
    meta = json.loads((rec / "session.json").read_text())
    sr, a = wavfile.read(str(rec / "band.wav"))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    dial = meta["dial_hz"]; guard = meta["guard_hz"]; width = meta["width_hz"]
    t_start, t_stop = meta["t_start"], meta["t_stop"]
    print(f"rec {rec.name}: {len(a)/sr:.0f}s @ {sr}Hz  dial {dial}  passband {guard:.0f}-{guard+width:.0f} Hz off dial")

    spots = load_spots(Path(args.spots), t_start, t_stop)
    print(f"spots inside window: {len(spots)}")

    # keep spots whose station lands in the recorded passband; strongest-first
    cand = []
    for s in spots:
        tone = abs(s["freq_khz"] * 1000.0 - dial)
        if not (guard <= tone <= guard + width):
            continue
        cand.append((s, tone))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mod, dmeta, sess = _B.load_deepcw()
    tr = int(dmeta["sample_rate"]); chars = list(dmeta["chars"]); blank = int(dmeta["blank_index"])
    inn, outn = dmeta["onnx_input_name"], dmeta["onnx_output_name"]

    def teacher(clip, csr):
        a2 = mod.resample_linear(clip.astype(np.float32), csr, tr)
        return mod.greedy_ctc_decode(sess.run([outn], {inn: mod.audio_to_spectrogram(a2, dmeta)})[0],
                                     chars, blank).strip()

    # group by call, enforce per-call spacing so we don't cut the same transmission twice
    by_call: dict[str, list] = {}
    for s, tone in sorted(cand, key=lambda x: x[0]["t"]):
        by_call.setdefault(s["call"], []).append((s, tone))

    rows = []
    idx = 0
    kept = skipped_short = skipped_tok = skipped_agree = skipped_weak = 0
    for call, items in by_call.items():
        last_t = -1e9
        # count distinct skimmers near this call/time for a strength gate
        for s, tone in items:
            if s["t"] - last_t < args.merge_s:
                continue
            n_sk = len({x[0]["spotter"] for x in items if abs(x[0]["t"] - s["t"]) <= 60})
            if n_sk < args.min_skimmers:
                skipped_weak += 1; continue
            i0 = int((s["t"] - args.pre - t_start) * sr)
            i1 = int((s["t"] + args.post - t_start) * sr)
            clip = isolate(a, sr, tone, i0, i1)
            if clip.size < sr:
                continue
            clip = clip / (np.abs(clip).max() + 1e-9) * 0.95
            label = teacher(clip, sr)
            if len([c for c in label if c != " "]) < args.min_chars:
                skipped_short += 1; continue
            if not tokenizes(label):
                skipped_tok += 1; continue
            agree = call in label.replace(" ", "")
            if args.require_agree and not agree:
                skipped_agree += 1; continue
            store = resample_poly(clip, STORE_SR, sr).astype(np.float32)
            store = store / (np.abs(store).max() + 1e-9) * 0.95
            fn = f"clip_{idx:05d}.wav"
            wavfile.write(str(out / fn), STORE_SR, (store * 32767).astype(np.int16))
            rows.append({"file": fn, "text": label,
                         "meta": {"snr": 99, "rbn_call": call, "agree": bool(agree),
                                  "skimmers": n_sk, "offset_hz": round(tone, 1),
                                  "t": round(s["t"] - t_start, 1), "src": rec.name}})
            kept += 1; idx += 1; last_t = s["t"]

    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"kept={kept}  skipped: weak={skipped_weak} short={skipped_short} "
          f"untokenizable={skipped_tok} disagree={skipped_agree}")
    agree_n = sum(1 for r in rows if r["meta"]["agree"])
    print(f"RBN-agreeing clips: {agree_n}/{kept}")
    for r in rows[:10]:
        print(f"  {r['file']} [{r['meta']['rbn_call']:8s} off{r['meta']['offset_hz']:.0f} "
              f"{'AGREE' if r['meta']['agree'] else '     '}]  {r['text']!r}")
    print(f"wrote {out/'labels.jsonl'}")


if __name__ == "__main__":
    main()
