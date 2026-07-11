import numpy as np
import torch
import onnxruntime as ort
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.export.to_onnx import export_onnx


def _run_onnx(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(["log_probs"], {"spectrogram": x.numpy()})[0]


def test_onnx_matches_pytorch_and_dynamic_time(tmp_path):
    net = CwCtcNet(time_downsample=2).eval()
    out = tmp_path / "m.onnx"
    export_onnx(net, str(out))
    for T in (751, 400):                       # dynamic-time check
        x = torch.randn(1, 1, 23, T)
        with torch.no_grad():
            torch_lp = net(x).numpy()
        onnx_lp = _run_onnx(str(out), x)
        assert onnx_lp.shape == torch_lp.shape
        assert np.allclose(onnx_lp, torch_lp, atol=1e-3, rtol=1e-3)
        assert greedy_ctc_decode(torch.from_numpy(onnx_lp)) == \
               greedy_ctc_decode(torch.from_numpy(torch_lp))


def test_export_requires_eval_mode(tmp_path):
    net = CwCtcNet().train()
    import pytest
    with pytest.raises(AssertionError):
        export_onnx(net, str(tmp_path / "x.onnx"))
