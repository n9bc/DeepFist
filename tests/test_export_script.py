import importlib.util
from pathlib import Path
import onnxruntime as ort
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_exports_and_runs(tmp_path):
    export = _load("export")
    from deepfist.model.net import CwCtcNet
    ckpt = tmp_path / "m.pt"
    torch.save(CwCtcNet().state_dict(), ckpt)
    out = tmp_path / "deepfist.onnx"
    export.run_export(str(ckpt), str(out), downsample=2)
    assert out.exists() and (out.parent / "deepfist.onnx.json").exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y = sess.run(["log_probs"], {"spectrogram": np.random.randn(1, 1, 23, 400).astype("float32")})[0]
    assert y.shape[2] == 48
