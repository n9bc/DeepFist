"""Export the CNN+CTC model to ONNX (spectrogram-in, dynamic batch+time) + metadata."""
import json
import os
import subprocess
from datetime import datetime, timezone

import torch

from deepfist.morse.alphabet import TOKENS
from deepfist.features.spectrogram import (
    N_FFT, HOP, BAND_LO_HZ, BAND_HI_HZ, SAMPLE_RATE, FREQ_BINS)
from deepfist.model.net import CwCtcNet


def export_onnx(model, out_path: str, example_time: int = 401, opset: int = 17) -> None:
    assert not model.training, "call model.eval() before export (folds BatchNorm)"
    example = torch.randn(1, 1, FREQ_BINS, example_time)
    torch.onnx.export(
        model, example, out_path,
        input_names=["spectrogram"], output_names=["log_probs"],
        dynamic_axes={"spectrogram": {0: "batch", 3: "time"},
                      "log_probs": {0: "time_out", 1: "batch"}},
        opset_version=opset,
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_metadata(onnx_path: str, time_downsample: int = 2) -> str:
    meta = {
        "model": "deepfist-cw-ctc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input": {"name": "spectrogram", "layout": f"[batch,1,freq={FREQ_BINS},time]",
                  "dtype": "float32"},
        "output": {"name": "log_probs", "layout": "[time_out,batch,class=48]",
                   "dtype": "float32"},
        "preprocessing": {
            "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP,
            "band_lo_hz": BAND_LO_HZ, "band_hi_hz": BAND_HI_HZ, "freq_bins": FREQ_BINS,
            "window": "hann", "center": True,
            "magnitude": "abs", "compress": "log1p", "normalize": "global_standardize",
        },
        "ctc": {"blank_index": 0, "time_downsample": time_downsample},
        "tokens": list(TOKENS),
    }
    meta_path = onnx_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def export_from_checkpoint(ckpt_path: str, out_path: str,
                           time_downsample: int = 2, opset: int = 17) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    net = CwCtcNet(time_downsample=time_downsample)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()
    export_onnx(net, out_path, opset=opset)
    write_metadata(out_path, time_downsample=time_downsample)
