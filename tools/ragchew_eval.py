"""Conversational-copy (rag-chew) CER eval — the metric for the plain-text/QSO half of the
goal that we were missing. Fixed cached set of rag-chew utterances (known text) rendered to
CW; reports mean CER for a checkpoint. Baseline exp16 ~0.225 (vs 0.031 on ARRL prose) —
proves the rag-chew gap. Re-run on exp22_ragchew to test whether rag-chew training closed it.

Synthetic (leakage caveat) but it is the ONLY labeled conversational-CER signal we have;
the real anchors are callsign-only, ARRL is prose. Run with DEEPFIST_CONDITION=1.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
EVAL = ROOT / "data" / "ragchew_eval"


def build(n=150, seed0=700000):
    """Fixed rag-chew clips (deterministic): render conversational utterances to CW."""
    import numpy as np
    from scipy.io import wavfile
    from deepfist.synth.generator import generate, GenConfig
    from deepfist.synth.text import _ragchew
    EVAL.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig(snr_range=(3.0, 18.0))          # moderate/clean so it measures COPY, not SNR
    rows, seed, kept = [], seed0, 0
    rng = np.random.default_rng(seed0)
    while kept < n:
        # force a rag-chew label, then render it deterministically via the generator's seed
        s = generate(seed=seed, config=cfg); seed += 1
        lbl = s.label.strip().upper()
        if "=" not in lbl and not any(t in lbl.split() for t in ("TNX", "UR", "RST", "QTH", "DE", "73")):
            continue
        fn = f"rc_{kept:04d}.wav"
        wavfile.write(str(EVAL / fn), cfg.sample_rate, np.asarray(s.audio, np.float32))
        rows.append({"file": fn, "text": lbl})
        kept += 1
    (EVAL / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def evaluate(ckpt):
    import numpy as np, torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
    from deepfist.features.conditioner import maybe_condition
    from deepfist.model.decode import greedy_ctc_decode
    from deepfist.train.metrics import cer
    import callsign_eval as CE
    net = CE._load_net(ckpt)
    rows = [json.loads(l) for l in (EVAL / "labels.jsonl").read_text().splitlines() if l.strip()]
    tot = 0.0
    with torch.no_grad():
        for r in rows:
            sr, a = wavfile.read(str(EVAL / r["file"]))
            if a.ndim > 1: a = a.mean(1)
            a = a.astype(np.float32) / (np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0)
            if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
            lp = net(audio_to_spectrogram(maybe_condition(a, SAMPLE_RATE), SAMPLE_RATE).unsqueeze(0).unsqueeze(0))
            tot += cer(greedy_ctc_decode(lp)[0], r["text"])
    return tot / len(rows), len(rows)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
    if not (EVAL / "labels.jsonl").exists():
        print("building rag-chew eval set..."); build()
    c, n = evaluate(ckpt)
    print(f"{ckpt}: rag-chew CER {c:.3f} = {100*(1-c):.1f}% char acc  ({n} clips)")


if __name__ == "__main__":
    main()
