import wave
import numpy as np
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_listen_writes_valid_wav(tmp_path):
    listen = _load("listen")
    out = tmp_path / "clip.wav"
    audio = (0.5 * np.sin(np.linspace(0, 100, 8000))).astype(np.float32)
    listen.write_wav(str(out), audio, 8000)
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 8000
        assert w.getnframes() == 8000
        assert w.getsampwidth() == 2
