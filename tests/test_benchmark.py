"""Unit tests for the DeepCW head-to-head benchmark harness (pure logic +
DeepCW front-end tensor shape). No training required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_vs_deepcw as B

EVAL_DIR = Path(__file__).resolve().parents[1] / "runs" / "wmr_evalA"


def test_snr_bucket_rounds():
    assert B.snr_bucket(-6.0) == -6
    assert B.snr_bucket(0.0) == 0
    assert B.snr_bucket(9.6) == 10


def test_score_perfect_and_aggregation():
    rows = [(None, "5NN", -6.0), (None, "K7CO", -6.0), (None, "TEST", 10.0)]
    # perfect predictions -> 0 CER everywhere
    r = B.score(["5NN", "K7CO", "TEST"], rows)
    assert r["n"] == 3 and r["overall"] == 0.0 and r["buckets"][-6] == 0.0 and r["buckets"][10] == 0.0
    # one wholly-wrong 4-char pred at +10 -> CER 1.0 in that bucket
    r2 = B.score(["5NN", "K7CO", "XXXX"], rows)
    assert r2["buckets"][-6] == 0.0 and r2["buckets"][10] == pytest.approx(1.0)


def test_score_space_normalized_ignores_word_spacing():
    # DeepCW-style run-together prediction should score 0 under the primary metric
    rows = [(None, "R 5NN NE TU", 10.0)]
    r = B.score(["R5NNNETU"], rows)
    assert r["overall"] == 0.0            # spaces stripped -> identical
    assert r["overall_raw"] > 0.0          # raw CER still penalizes the 3 missing spaces


def test_surpasses_requires_overall_and_all_high_snr_buckets():
    dcw_ov, dcw_b = 0.12, {-6: 0.20, 0: 0.10, 6: 0.05}
    # better overall AND <= at every bucket >= 0 dB -> wins
    assert B.surpasses(0.10, {-6: 0.30, 0: 0.09, 6: 0.05}, dcw_ov, dcw_b) is True
    # better overall but worse at +6 dB -> loses
    assert B.surpasses(0.10, {-6: 0.10, 0: 0.09, 6: 0.06}, dcw_ov, dcw_b) is False
    # tie/worse overall -> loses
    assert B.surpasses(0.12, {-6: 0.01, 0: 0.01, 6: 0.01}, dcw_ov, dcw_b) is False


@pytest.mark.skipif(not EVAL_DIR.exists(), reason="wmr_evalA not generated")
def test_load_eval_reads_labels():
    rows = B.load_eval(EVAL_DIR)
    assert len(rows) > 0
    wav, text, snr = rows[0]
    assert Path(wav).suffix == ".wav" and isinstance(text, str) and -7 <= snr <= 11


@pytest.mark.skipif(not (B.DEEPCW_DIR / "model.onnx").exists() or not EVAL_DIR.exists(),
                    reason="DeepCW install or wmr_evalA missing")
def test_deepcw_frontend_shape_matches_onnx_spec():
    mod, meta, _sess = B.load_deepcw()
    rows = B.load_eval(EVAL_DIR)[:1]
    srate, audio = mod.read_wav_mono(Path(rows[0][0]))
    audio = mod.resample_linear(audio, srate, int(meta["sample_rate"]))
    spec = mod.audio_to_spectrogram(audio, meta)
    # DeepCW input layout is [batch, channel, time, frequency]
    assert spec.ndim == 4 and spec.shape[0] == 1 and spec.shape[1] == 1
    assert spec.shape[3] == int(meta["spectrogram_frequency_bins"]) == 65
    assert spec.dtype == np.float32
