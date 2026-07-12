import json
from pathlib import Path
import numpy as np
import pytest
import torch
import onnxruntime as ort
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.export.to_onnx import (export_onnx, write_metadata,
                                     export_from_checkpoint)
from deepfist.morse.alphabet import TOKENS


def _run_onnx(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(["log_probs"], {"spectrogram": x.numpy()})[0]


def test_onnx_matches_pytorch_and_dynamic_time(tmp_path):
    net = CwCtcNet(time_downsample=2).eval()
    out = tmp_path / "m.onnx"
    export_onnx(net, str(out))
    for T in (401, 200):                       # dynamic-time check
        x = torch.randn(1, 1, 65, T)
        with torch.no_grad():
            torch_lp = net(x).numpy()
        onnx_lp = _run_onnx(str(out), x)
        assert onnx_lp.shape == torch_lp.shape
        assert np.allclose(onnx_lp, torch_lp, atol=1e-3, rtol=1e-3)
        assert greedy_ctc_decode(torch.from_numpy(onnx_lp)) == \
               greedy_ctc_decode(torch.from_numpy(torch_lp))


def test_export_requires_eval_mode(tmp_path):
    net = CwCtcNet().train()
    with pytest.raises(AssertionError):
        export_onnx(net, str(tmp_path / "x.onnx"))


def test_metadata_sidecar(tmp_path):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"stub")
    meta_path = write_metadata(str(onnx_path), time_downsample=2)
    meta = json.loads(Path(meta_path).read_text())
    assert meta["tokens"] == TOKENS and len(meta["tokens"]) == 48
    assert meta["ctc"]["blank_index"] == 0
    assert meta["preprocessing"]["n_fft"] == 256
    assert meta["preprocessing"]["freq_bins"] == 65


def test_export_from_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_from_checkpoint(str(tmp_path / "nope.pt"), str(tmp_path / "o.onnx"))
