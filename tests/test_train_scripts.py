import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_runs_on_untrained_model():
    ev = _load("evaluate")
    from deepfist.model.net import CwCtcNet
    net = CwCtcNet()
    table = ev.run_eval(net, snr_points=[6, 0], clips_per_point=3, device="cpu")
    assert set(table.keys()) == {6.0, 0.0}
    assert all(0.0 <= v for v in table.values())
