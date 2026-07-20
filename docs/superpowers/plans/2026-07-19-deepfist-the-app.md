# DeepFistTheApp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Chosen execution mode for this run:** the user chose continued inline work in
> this same Claude Desktop session — execution will use
> **superpowers:executing-plans** (batch execution with checkpoints).

**Goal:** Build DeepFistTheApp — a cross-platform PySide6 desktop CW reader that decodes system-output audio (or a WAV) through the existing DeepFist pipeline via ONNX Runtime, per the approved spec `docs/superpowers/specs/2026-07-19-deepfist-the-app-design.md`.

**Architecture:** A new sibling package `deepfistapp/` (audio sources → bounded queue → QThread decoder worker → Qt signals → themed UI) reuses `deepfist.*` DSP untouched; `tools/squelch|despike|tempo` are first promoted verbatim into `deepfist/dsp/` with re-export shims. Inference is ONNX Runtime with a numpy spectrogram port gated by parity tests; torch remains a dev/test-only dependency.

**Tech Stack:** Python 3.11+, PySide6, onnxruntime, numpy, scipy, soundcard (loopback capture), pytest + pytest-qt, PyInstaller.

## Global Constraints

Copied from the approved spec — every task implicitly includes these:

- License **GPL-3.0-or-later**; platforms **Windows, Linux, macOS**.
- Fixed processing order: `raw-audio squelch → despike → condition → spectrogram → ONNX inference`. Conditioning never runs before the squelch.
- **All input is converted to mono 3200 Hz** (`SAMPLE_RATE = 3200`) before any processing.
- **No signal energy → no characters.** A closed squelch gate skips despike/condition/spectrogram/inference entirely and never emits text.
- Display defaults: strip `,` and `.`; BT (`=`) rendered as a line break; blank penalty **0**.
- Engine defaults: `tick = 0.4 s`, `guard = 1.3 s`, `window = 6.0 s`, squelch threshold `DEFAULT_THRESH = 12.0`, queue bound **32 blocks, drop-oldest**.
- Model contract: input `spectrogram` `[batch,1,65,time]` float32; output `log_probs` `[time_out,batch,48]`; blank index 0; `len(TOKENS) == 48`.
- **Nothing in `deepfistapp` imports torch.** App runtime deps exactly: `PySide6`, `onnxruntime`, `numpy`, `scipy`, `soundcard`.
- Import boundaries: `deepfistapp.engine` may import `deepfist.*`; `deepfistapp.ui` imports `deepfistapp.engine` but never `deepfist.*` directly; `deepfist.*` never imports `deepfistapp.*`.
- Theme tokens copied **exactly** from the spec §7.2 table (source: Lyra `src/theme.h`); glass only for overlay chips + settings drawer backdrop.
- Promotion of `tools/squelch.py`, `tools/despike.py`, `tools/tempo.py` into `deepfist/dsp/` is **semantics-preserving** (verbatim move + shims); the pre-existing test suite (109 tests) passes **unmodified** at every commit.
- numpy spectrogram parity with the torch reference: max abs diff ≤ 1e-4; golden-WAV decoded text **string-identical** between app engine and torch reference.
- Settings persist via `QSettings("DeepFist", "DeepFistTheApp")`.
- All commands below run from the repo root `C:\dev\DeepFist` in Git Bash; the interpreter is `.venv/Scripts/python.exe`.

## Complete File Map

**Modified (existing):**
- `pyproject.toml` — `app` extra, `pytest-qt` dev dep, gui-script, package find list
- `tools/squelch.py`, `tools/despike.py`, `tools/tempo.py` — become re-export shims

**Created — library promotion:**
- `deepfist/dsp/__init__.py`, `deepfist/dsp/squelch.py`, `deepfist/dsp/despike.py`, `deepfist/dsp/tempo.py`
- Test: `tests/dsp/__init__.py`, `tests/dsp/test_promotion.py`

**Created — app engine:**
- `deepfistapp/__init__.py`, `deepfistapp/theme.py`, `deepfistapp/config.py`, `deepfistapp/main.py`
- `deepfistapp/engine/__init__.py`, `numpy_features.py`, `commit.py`, `states.py`, `session.py`, `pipeline.py`, `spectra.py`, `worker.py`
- `deepfistapp/audio/__init__.py`, `source.py`, `wav_source.py`, `devices.py`, `loopback.py`
- `deepfistapp/resources/macos_loopback.md` (guide text; the `.onnx` is a build-time artifact, never committed)

**Created — app UI:**
- `deepfistapp/ui/__init__.py`, `main_window.py`, `waterfall.py`, `spectrum.py`, `meters.py`, `transcript.py`, `stats_bar.py`, `settings_drawer.py`, `transport.py`, `banners.py`

**Created — app tests (`tests/app/`):**
- `__init__.py`, `conftest.py`, `test_numpy_features.py`, `test_commit.py`, `test_states.py`, `test_theme.py`, `test_config.py`, `test_wav_source.py`, `test_devices.py`, `test_session.py`, `test_pipeline.py`, `test_spectra.py`, `test_worker.py`, `test_ui_main.py`, `test_ui_panels.py`, `test_ui_settings.py`, `test_integration.py`

**Created — packaging:**
- `packaging/deepfist_app.spec`, `packaging/build_release.py`

---

### Task 1: Promote squelch/despike/tempo into `deepfist/dsp/` with shims

**Files:**
- Create: `deepfist/dsp/__init__.py`, `deepfist/dsp/squelch.py`, `deepfist/dsp/despike.py`, `deepfist/dsp/tempo.py`
- Modify: `tools/squelch.py`, `tools/despike.py`, `tools/tempo.py` (reduce to shims)
- Test: `tests/dsp/__init__.py`, `tests/dsp/test_promotion.py`

**Interfaces:**
- Consumes: existing `tools/*.py` contents.
- Produces (used by every later engine task):
  - `deepfist.dsp.squelch.has_signal(audio: np.ndarray, sr: int, thresh: float = DEFAULT_THRESH) -> tuple[bool, float]`; `DEFAULT_THRESH = 12.0`; `keying_ratio(audio, sr) -> float`; `calibrate(dead_air, sr, margin=3.0, win_s=6.0, hop_s=3.0) -> tuple[float, float]`
  - `deepfist.dsp.despike.despike(x: np.ndarray, sr: int, k: float = 5.0, win_ms: float = 8.0, guard_ms: float = 0.5) -> np.ndarray`; `spike_fraction(x, sr, **kw) -> float`
  - `deepfist.dsp.tempo.estimate_wpm(x: np.ndarray, sr: int) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/dsp/__init__.py` (empty) and `tests/dsp/test_promotion.py`:

```python
"""Promotion equivalence: deepfist.dsp modules ARE the old tools modules."""
import sys
from pathlib import Path

import numpy as np

TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def test_squelch_shim_is_same_object():
    import squelch as shim
    from deepfist.dsp import squelch as pkg
    assert shim.has_signal is pkg.has_signal
    assert shim.keying_ratio is pkg.keying_ratio
    assert shim.calibrate is pkg.calibrate
    assert shim.DEFAULT_THRESH == pkg.DEFAULT_THRESH == 12.0


def test_despike_shim_is_same_object():
    import despike as shim
    from deepfist.dsp import despike as pkg
    assert shim.despike is pkg.despike
    assert shim.spike_fraction is pkg.spike_fraction


def test_tempo_shim_is_same_object():
    import tempo as shim
    from deepfist.dsp import tempo as pkg
    assert shim.estimate_wpm is pkg.estimate_wpm


def test_squelch_behavior_unchanged():
    rng = np.random.default_rng(0)
    sr = 3200
    t = np.arange(6 * sr) / sr
    # keyed tone: 700 Hz gated on/off at ~10 Hz -> high keying ratio
    gate = (np.sin(2 * np.pi * 5 * t) > 0).astype(np.float32)
    keyed = (np.sin(2 * np.pi * 700 * t) * gate).astype(np.float32)
    steady = np.sin(2 * np.pi * 700 * t).astype(np.float32) \
        + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
    from deepfist.dsp.squelch import has_signal
    assert has_signal(keyed, sr)[0] is True
    assert has_signal(steady, sr)[0] is False


def test_despike_neutral_on_clean_tone():
    sr = 3200
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 700 * t).astype(np.float32)
    from deepfist.dsp.despike import despike
    np.testing.assert_allclose(despike(x, sr), x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/dsp/test_promotion.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfist.dsp'`

- [ ] **Step 3: Perform the move**

```bash
mkdir -p deepfist/dsp
git mv tools/squelch.py deepfist/dsp/squelch.py
git mv tools/despike.py deepfist/dsp/despike.py
git mv tools/tempo.py deepfist/dsp/tempo.py
```

Create `deepfist/dsp/__init__.py`:

```python
"""Promoted live-decode DSP gates (moved verbatim from tools/; see spec §3.1)."""
```

Create the three shims. `tools/squelch.py`:

```python
"""Shim — moved to deepfist.dsp.squelch (spec §3.1). Kept so existing
sys.path-based imports (scripts/tci_decode.py, eval tools) keep working."""
from deepfist.dsp.squelch import *          # noqa: F401,F403
from deepfist.dsp.squelch import (          # noqa: F401
    has_signal, keying_ratio, calibrate, DEFAULT_THRESH, FRAME_MS, CW_LO, CW_HI)
```

`tools/despike.py`:

```python
"""Shim — moved to deepfist.dsp.despike (spec §3.1)."""
from deepfist.dsp.despike import *          # noqa: F401,F403
from deepfist.dsp.despike import despike, spike_fraction   # noqa: F401
```

`tools/tempo.py`:

```python
"""Shim — moved to deepfist.dsp.tempo (spec §3.1)."""
from deepfist.dsp.tempo import *            # noqa: F401,F403
from deepfist.dsp.tempo import estimate_wpm, normalize     # noqa: F401
```

Do **not** edit the moved files' bodies at all (semantics-preserving).

- [ ] **Step 4: Run tests to verify pass, including full regression**

Run: `.venv/Scripts/python.exe -m pytest tests/dsp/test_promotion.py -q`
Expected: `5 passed`

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, 0 failures (baseline 109 + 5 new = `114 passed`)

Run: `.venv/Scripts/python.exe scripts/tci_decode.py --help`
Expected: exits 0 and prints the argparse usage (shim import path works).

- [ ] **Step 5: Commit**

```bash
git add deepfist/dsp tools/squelch.py tools/despike.py tools/tempo.py tests/dsp
git commit -m "refactor(dsp): promote squelch/despike/tempo into deepfist.dsp with shims"
```

---

### Task 2: App package scaffold + packaging metadata + app-test conftest

**Files:**
- Create: `deepfistapp/__init__.py`, `deepfistapp/audio/__init__.py`, `deepfistapp/engine/__init__.py`, `deepfistapp/ui/__init__.py`, `deepfistapp/resources/macos_loopback.md`
- Modify: `pyproject.toml`
- Test: `tests/app/__init__.py`, `tests/app/conftest.py`, `tests/app/test_scaffold.py`

**Interfaces:**
- Produces: importable `deepfistapp` package; `pip install -e ".[app]"` target; every later `tests/app/*` file relies on `conftest.py` for offscreen Qt + dependency skips.

- [ ] **Step 1: Write the failing test**

Create `tests/app/__init__.py` (empty), `tests/app/conftest.py`:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# App tests require the app extras; skip the whole directory cleanly when absent
pytest.importorskip("PySide6")
pytest.importorskip("onnxruntime")
```

Create `tests/app/test_scaffold.py`:

```python
import importlib

import deepfistapp


def test_subpackages_import():
    for mod in ("deepfistapp.audio", "deepfistapp.engine", "deepfistapp.ui"):
        importlib.import_module(mod)


def test_no_torch_in_app_imports():
    import sys
    torch_seen_before = "torch" in sys.modules
    importlib.import_module("deepfistapp")
    importlib.import_module("deepfistapp.audio")
    importlib.import_module("deepfistapp.engine")
    if not torch_seen_before:
        assert "torch" not in sys.modules, "deepfistapp must not import torch"


def test_macos_guide_resource_exists():
    from pathlib import Path
    guide = Path(deepfistapp.__file__).parent / "resources" / "macos_loopback.md"
    assert guide.exists() and "BlackHole" in guide.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_scaffold.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp'` (or skip if PySide6 missing → first run `.venv/Scripts/python.exe -m pip install PySide6 onnxruntime soundcard pytest-qt`)

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/__init__.py`:

```python
"""DeepFistTheApp — desktop CW reader on the DeepFist pipeline (GPL-3.0-or-later)."""
__version__ = "0.1.0"
APP_NAME = "DeepFistTheApp"
ORG_NAME = "DeepFist"
```

Create empty `deepfistapp/audio/__init__.py`, `deepfistapp/engine/__init__.py`, `deepfistapp/ui/__init__.py` (docstring-only).

Create `deepfistapp/resources/macos_loopback.md` with the spec §12 guide verbatim:

```markdown
# Capturing system audio on macOS

macOS has no built-in loopback. One-time setup:

1. **Install a virtual audio driver** — [BlackHole 2ch](https://existential.audio/blackhole/)
   (free) or Loopback.app (commercial). DeepFistTheApp never bundles these;
   install them yourself.
2. **Create a Multi-Output Device** in *Audio MIDI Setup* containing both your
   speakers and BlackHole, so audio stays audible while being captured.
3. **Set the Multi-Output Device as the system output** (Sound settings).
4. In DeepFistTheApp, **select "BlackHole 2ch" as the source**.

## Troubleshooting

- *No devices listed* → the virtual driver is not installed (step 1).
- *Meters stay at zero / silence* → system output is not routed through the
  Multi-Output Device (step 3).
```

Modify `pyproject.toml` — the three touched sections become exactly:

```toml
[project.optional-dependencies]
audio = ["sounddevice>=0.4"]   # soundcard live decode (scripts/live_decode.py)
dev = ["pytest>=8.0", "pytest-qt>=4.4", "onnx>=1.22", "onnxruntime>=1.27"]
app = ["PySide6>=6.7", "onnxruntime>=1.27", "soundcard>=0.4"]

[project.gui-scripts]
deepfist-app = "deepfistapp.main:main"

[tool.setuptools.packages.find]
include = ["deepfist*", "deepfistapp*"]
```

(`deepfistapp.main` does not exist until Task 17; the gui-script entry is
declared now so packaging metadata is complete — it is not invoked until then.)

Then install the extras: `.venv/Scripts/python.exe -m pip install -e ".[app,dev]"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_scaffold.py -q`
Expected: `3 passed`

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, 0 failures (`117 passed`)

- [ ] **Step 5: Commit**

```bash
git add deepfistapp pyproject.toml tests/app
git commit -m "feat(app): DeepFistTheApp package scaffold, app extras, test conftest"
```

---

### Task 3: numpy spectrogram port with torch parity gate

**Files:**
- Create: `deepfistapp/engine/numpy_features.py`
- Test: `tests/app/test_numpy_features.py`

**Interfaces:**
- Consumes: constants mirrored from `deepfist/features/spectrogram.py` (`N_FFT=256, HOP=48, BAND_LO_HZ=400, BAND_HI_HZ=1200, SAMPLE_RATE=3200, FREQ_BINS=65`).
- Produces: `audio_to_spectrogram_np(audio: np.ndarray, sample_rate: int = 3200) -> np.ndarray` — float32 `[65, T]`, `T = 1 + len(audio)//48`; module constants `SAMPLE_RATE`, `FREQ_BINS`, `N_FFT`, `HOP`. Used by Tasks 10, 17.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_numpy_features.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # parity reference only — dev env


def _make_signals():
    rng = np.random.default_rng(7)
    sr = 3200
    t = np.arange(6 * sr) / sr
    gate = (np.sin(2 * np.pi * 4 * t) > 0).astype(np.float32)
    return sr, [
        (np.sin(2 * np.pi * 600 * t) * gate).astype(np.float32),          # clean keyed
        (np.sin(2 * np.pi * 700 * t) * gate
         + 0.3 * rng.standard_normal(len(t))).astype(np.float32),          # noisy keyed
        rng.standard_normal(len(t)).astype(np.float32) * 0.1,              # noise only
    ]


def test_constants_match_torch_module():
    from deepfist.features import spectrogram as ref
    from deepfistapp.engine import numpy_features as npf
    assert (npf.N_FFT, npf.HOP) == (ref.N_FFT, ref.HOP)
    assert (npf.SAMPLE_RATE, npf.FREQ_BINS) == (ref.SAMPLE_RATE, ref.FREQ_BINS)


def test_parity_within_1e_4():
    from deepfist.features.spectrogram import audio_to_spectrogram
    from deepfistapp.engine.numpy_features import audio_to_spectrogram_np
    sr, signals = _make_signals()
    for sig in signals:
        ref = audio_to_spectrogram(sig, sr).numpy()
        got = audio_to_spectrogram_np(sig, sr)
        assert got.shape == ref.shape == (65, 1 + len(sig) // 48)
        assert got.dtype == np.float32
        assert float(np.max(np.abs(got - ref))) <= 1e-4


def test_numpy_features_does_not_import_torch():
    import subprocess, sys
    code = ("import sys; import deepfistapp.engine.numpy_features; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code])
    assert r.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_numpy_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.numpy_features'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/numpy_features.py`:

```python
"""numpy port of deepfist.features.spectrogram (torch-free app runtime).

Constants are mirrored, not imported: the torch module imports torch at module
level, and nothing in deepfistapp may import torch. The parity test pins both
the constants and the numeric output against the torch reference.
"""
import numpy as np

N_FFT = 256
HOP = 48
BAND_LO_HZ = 400
BAND_HI_HZ = 1200
SAMPLE_RATE = 3200

_HZ_PER_BIN = SAMPLE_RATE / N_FFT
_LO = int(np.ceil(BAND_LO_HZ / _HZ_PER_BIN))            # 32
_HI = int(np.floor(BAND_HI_HZ / _HZ_PER_BIN)) + 1       # 97 (exclusive)
FREQ_BINS = _HI - _LO                                    # 65

# torch.hann_window default is PERIODIC: 0.5*(1-cos(2*pi*n/N)) — np.hanning is
# symmetric and would break parity.
_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float64)


def audio_to_spectrogram_np(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    pad = N_FFT // 2
    xp = np.pad(x, pad, mode="reflect")                  # torch.stft center=True
    n_frames = 1 + len(x) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * _WINDOW                           # [T, N_FFT]
    mag = np.abs(np.fft.rfft(frames, axis=1)).T          # [freq, T]
    spec = np.log1p(mag[_LO:_HI])                        # [65, T]
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_numpy_features.py -q`
Expected: `3 passed`. If the parity assertion fails, the divergence is a bug in
this port (window periodicity and reflect padding are the usual culprits) — fix
the port; changing the tolerance is not allowed (spec §3.3).

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/numpy_features.py tests/app/test_numpy_features.py
git commit -m "feat(app): torch-free numpy spectrogram with 1e-4 parity gate"
```

---

### Task 4: Commit logic + display transforms (`engine/commit.py`)

**Files:**
- Create: `deepfistapp/engine/commit.py`
- Test: `tests/app/test_commit.py`

**Interfaces:**
- Consumes: `deepfist.morse.alphabet.TOKENS` (pure-python, no torch).
- Produces (used by Tasks 10, 11, 17):
  - `BLANK_ID = 0`
  - `greedy_frames(log_probs: np.ndarray, blank_pen: float = 0.0) -> tuple[list[int], list[int], int]` — input `[T, 1, 48]`; returns (char ids, frame indices, T)
  - `commit_new(ids: list[int], frames: list[int], n_frames: int, *, window_s: float, audio_end_s: float, committed_t: float, guard_s: float) -> tuple[list[int], float]` — ids to emit now + advanced boundary (verbatim port of `scripts/tci_decode.py` commit rule)
  - `render_text(ids: list[int], strip_punct: bool = True, bt_mode: str = "newline") -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_commit.py`:

```python
import numpy as np

from deepfist.morse.alphabet import TOKENS


def _logprobs_for(char_frame_pairs, n_frames):
    """[T,1,48] log-probs: blank everywhere except the given (token, frame)s."""
    lp = np.full((n_frames, 1, len(TOKENS)), -10.0, dtype=np.float32)
    lp[:, 0, 0] = -0.1                       # blank wins by default
    for tok, fr in char_frame_pairs:
        lp[fr, 0, 0] = -10.0
        lp[fr, 0, TOKENS.index(tok)] = -0.1
    return lp


def test_greedy_frames_collapses_and_reports_frames():
    from deepfistapp.engine.commit import greedy_frames
    lp = _logprobs_for([("C", 10), ("Q", 20)], n_frames=40)
    ids, frames, T = greedy_frames(lp)
    assert [TOKENS[i] for i in ids] == ["C", "Q"]
    assert frames == [10, 20] and T == 40


def test_commit_emits_each_char_exactly_once_across_overlapping_windows():
    from deepfistapp.engine.commit import greedy_frames, commit_new
    # 6 s window, char at t=3.0 s -> frame 20/40. First decode: not settled
    # (guard 1.3 puts settle at 4.7 but char must ALSO be past committed_t).
    lp = _logprobs_for([("K", 20)], n_frames=40)
    ids, frames, T = greedy_frames(lp)
    emit1, c1 = commit_new(ids, frames, T, window_s=6.0, audio_end_s=6.0,
                           committed_t=0.0, guard_s=1.3)
    assert [TOKENS[i] for i in emit1] == ["K"]      # 3.0 <= 4.7 -> emitted
    # Re-decode 0.4 s later: same char now at 2.6 s absolute-in-window; the
    # boundary c1=4.7 already covers it -> must NOT re-emit.
    lp2 = _logprobs_for([("K", 17)], n_frames=40)   # 6.4-6.0+... same audio time
    ids2, frames2, T2 = greedy_frames(lp2)
    emit2, c2 = commit_new(ids2, frames2, T2, window_s=6.0, audio_end_s=6.4,
                           committed_t=c1, guard_s=1.3)
    assert emit2 == [] and c2 == 6.4 - 1.3


def test_render_text_defaults():
    from deepfistapp.engine.commit import render_text
    ids = [TOKENS.index(c) for c in ["C", "Q", " ", "=", "A", ",", "."]]
    assert render_text(ids) == "CQ \nA"
    assert render_text(ids, strip_punct=False, bt_mode="eq") == "CQ =A,."
    assert render_text(ids, bt_mode="prosign") == "CQ <BT>A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_commit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.commit'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/commit.py`:

```python
"""Greedy CTC frame decode + settled-time commit rule + display transforms.

commit_new is a verbatim port of the boundary logic in scripts/tci_decode.py
(lines around `settle_to`): emit a char once its audio time falls in
(committed_t, audio_end - guard]; advance the boundary by settled TIME, never
by char timestamps, so re-decodes of the overlapping window never reprint.
"""
import numpy as np

from deepfist.morse.alphabet import TOKENS

BLANK_ID = 0


def greedy_frames(log_probs: np.ndarray, blank_pen: float = 0.0):
    lp = np.array(log_probs, dtype=np.float32, copy=True)
    lp[..., BLANK_ID] -= blank_pen
    args = lp.argmax(-1)[:, 0].tolist()
    prev, ids, frames = None, [], []
    for t, s in enumerate(args):
        if s != prev:
            if s != BLANK_ID:
                ids.append(int(s))
                frames.append(t)
            prev = s
    return ids, frames, len(args)


def commit_new(ids, frames, n_frames, *, window_s, audio_end_s, committed_t, guard_s):
    win_start = audio_end_s - window_s
    settle_to = audio_end_s - guard_s
    emit = [cid for cid, fr in zip(ids, frames)
            if committed_t < win_start + (fr / max(1, n_frames)) * window_s <= settle_to]
    return emit, max(committed_t, settle_to)


def render_text(ids, strip_punct: bool = True, bt_mode: str = "newline") -> str:
    text = "".join(TOKENS[i] for i in ids)
    if bt_mode == "newline":
        text = text.replace("=", "\n")
    elif bt_mode == "prosign":
        text = text.replace("=", "<BT>")
    if strip_punct:
        text = text.replace(",", "").replace(".", "")
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_commit.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/commit.py tests/app/test_commit.py
git commit -m "feat(app): CTC frame decode, settled-commit rule, display transforms"
```

---

### Task 5: State machine (`engine/states.py`)

**Files:**
- Create: `deepfistapp/engine/states.py`
- Test: `tests/app/test_states.py`

**Interfaces:**
- Produces (used by Tasks 13, 17):
  - `class AppState(Enum): IDLE, LISTENING, PLAYING, PAUSED, DEVICE_LOST, RESCANNING`
  - `class IllegalTransition(RuntimeError)`
  - `class StateMachine:` with `state: AppState` (starts `IDLE`) and `to(new: AppState) -> AppState` (raises `IllegalTransition` on any pair not in spec §8)

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_states.py`:

```python
import pytest


def test_legal_paths_from_spec_section_8():
    from deepfistapp.engine.states import AppState as S, StateMachine
    sm = StateMachine()
    assert sm.state is S.IDLE
    for step in (S.LISTENING, S.DEVICE_LOST, S.RESCANNING, S.LISTENING, S.IDLE,
                 S.PLAYING, S.PAUSED, S.PLAYING, S.IDLE):
        sm.to(step)
    assert sm.state is S.IDLE
    sm.to(S.LISTENING); sm.to(S.DEVICE_LOST); sm.to(S.LISTENING)   # direct re-pick
    sm.to(S.DEVICE_LOST); sm.to(S.RESCANNING); sm.to(S.DEVICE_LOST)


def test_illegal_transitions_raise():
    from deepfistapp.engine.states import AppState as S, StateMachine, IllegalTransition
    for start, bad in [(S.IDLE, S.PAUSED), (S.IDLE, S.DEVICE_LOST),
                       (S.PLAYING, S.LISTENING), (S.PAUSED, S.LISTENING),
                       (S.LISTENING, S.PLAYING), (S.RESCANNING, S.IDLE)]:
        sm = StateMachine(); sm.state = start
        with pytest.raises(IllegalTransition):
            sm.to(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_states.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.states'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/states.py`:

```python
"""App/session state machine — exactly the transitions in spec §8."""
from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PLAYING = auto()
    PAUSED = auto()
    DEVICE_LOST = auto()
    RESCANNING = auto()


class IllegalTransition(RuntimeError):
    pass


_S = AppState
LEGAL: set[tuple[AppState, AppState]] = {
    (_S.IDLE, _S.LISTENING), (_S.LISTENING, _S.IDLE),
    (_S.IDLE, _S.PLAYING), (_S.PLAYING, _S.PAUSED), (_S.PAUSED, _S.PLAYING),
    (_S.PLAYING, _S.IDLE), (_S.PAUSED, _S.IDLE),
    (_S.LISTENING, _S.DEVICE_LOST),
    (_S.DEVICE_LOST, _S.RESCANNING), (_S.RESCANNING, _S.LISTENING),
    (_S.RESCANNING, _S.DEVICE_LOST), (_S.DEVICE_LOST, _S.LISTENING),
}


class StateMachine:
    def __init__(self) -> None:
        self.state = AppState.IDLE

    def to(self, new: AppState) -> AppState:
        if (self.state, new) not in LEGAL:
            raise IllegalTransition(f"{self.state.name} -> {new.name}")
        self.state = new
        return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_states.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/states.py tests/app/test_states.py
git commit -m "feat(app): spec-exact session state machine"
```

---

### Task 6: Theme tokens + QSS (`theme.py`)

**Files:**
- Create: `deepfistapp/theme.py`
- Test: `tests/app/test_theme.py`

**Interfaces:**
- Produces (used by all UI tasks): module constants `BG_APP, BG_PANEL, BG_RECESS, BG_CTRL, ACCENT, ACCENT_DIM, ACCENT2, TEXT_PRIMARY, TEXT_MUTED, TEXT_FAINT, BORDER, ENGAGED_ORANGE, SELECTION, GLASS_PANEL, GLASS_BORDER, GLASS_SURFACE, TEXT_ON_GLASS, TEXT_ON_GLASS_BRIGHT, OUTLINE_SOFT, OUTLINE_HARD, DIAL_MARKER, RX_MARKER` (all `str`), and `build_qss() -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_theme.py`:

```python
EXACT = {  # spec §7.2 table — derived from Lyra src/theme.h, never approximated
    "BG_APP": "rgb(10,13,18)", "BG_PANEL": "rgb(17,22,32)",
    "BG_RECESS": "rgb(16,20,28)", "BG_CTRL": "rgb(22,28,40)",
    "ACCENT": "rgb(0,229,255)", "ACCENT_DIM": "rgb(0,168,200)",
    "ACCENT2": "rgb(57,255,20)", "TEXT_PRIMARY": "rgb(205,217,229)",
    "TEXT_MUTED": "rgb(138,154,172)", "TEXT_FAINT": "rgb(90,112,128)",
    "BORDER": "rgb(30,42,58)", "ENGAGED_ORANGE": "rgb(255,154,60)",
    "SELECTION": "rgba(0,229,255,80)",
    "GLASS_PANEL": "#cc14202a", "GLASS_BORDER": "#2a4a5a",
    "GLASS_SURFACE": "#101820", "TEXT_ON_GLASS": "#8fa6ba",
    "TEXT_ON_GLASS_BRIGHT": "#8fd0ff", "OUTLINE_SOFT": "#80000000",
    "OUTLINE_HARD": "#cc000000", "DIAL_MARKER": "#ffaa50",
    "RX_MARKER": "#a6ff00",
}


def test_tokens_are_exact():
    from deepfistapp import theme
    for name, value in EXACT.items():
        assert getattr(theme, name) == value, name


def test_qss_uses_tokens_and_styles_core_widgets():
    from deepfistapp.theme import build_qss
    qss = build_qss()
    for fragment in ("rgb(10,13,18)", "rgb(0,229,255)", "QMainWindow", "QPushButton",
                     "QMenuBar", "QSlider", "QComboBox", "QPlainTextEdit"):
        assert fragment in qss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_theme.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.theme'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/theme.py`:

```python
"""Lyra theme tokens — exact values from Lyra-SDR-cpp/src/theme.h (spec §7.2).
Re-derive from that file on theme changes; never hand-tweak here."""

BG_APP = "rgb(10,13,18)"
BG_PANEL = "rgb(17,22,32)"
BG_RECESS = "rgb(16,20,28)"
BG_CTRL = "rgb(22,28,40)"
ACCENT = "rgb(0,229,255)"
ACCENT_DIM = "rgb(0,168,200)"
ACCENT2 = "rgb(57,255,20)"
TEXT_PRIMARY = "rgb(205,217,229)"
TEXT_MUTED = "rgb(138,154,172)"
TEXT_FAINT = "rgb(90,112,128)"
BORDER = "rgb(30,42,58)"
ENGAGED_ORANGE = "rgb(255,154,60)"
SELECTION = "rgba(0,229,255,80)"
GLASS_PANEL = "#cc14202a"
GLASS_BORDER = "#2a4a5a"
GLASS_SURFACE = "#101820"
TEXT_ON_GLASS = "#8fa6ba"
TEXT_ON_GLASS_BRIGHT = "#8fd0ff"
OUTLINE_SOFT = "#80000000"
OUTLINE_HARD = "#cc000000"
DIAL_MARKER = "#ffaa50"
RX_MARKER = "#a6ff00"


def build_qss() -> str:
    return f"""
QMainWindow, QWidget {{ background: {BG_APP}; color: {TEXT_PRIMARY}; }}
QLabel {{ color: {TEXT_MUTED}; background: transparent; }}
QToolTip {{ background: {BG_PANEL}; color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT}; border-radius: 4px; padding: 8px 10px; }}
QLineEdit, QComboBox {{ background: {BG_CTRL}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; border-radius: 3px; padding: 4px 6px;
    selection-background-color: {SELECTION}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {BG_PANEL}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; selection-background-color: {SELECTION}; }}
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {BG_CTRL}, stop:1 {BG_RECESS});
    color: {TEXT_PRIMARY}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 12px; font-weight: 600; letter-spacing: 0.5px; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:checked {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {ACCENT_DIM}, stop:1 {BG_CTRL});
    border-color: {ACCENT}; color: {BG_APP}; }}
QPushButton:pressed {{ background: {BG_RECESS}; }}
QPushButton:disabled {{ background: {BG_RECESS}; color: {TEXT_FAINT};
    border-color: {BORDER}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BG_RECESS};
    border-radius: 2px; border: 1px solid {BORDER}; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px;
    margin: -6px 0; border-radius: 2px; border: 1px solid {BG_APP}; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT2}; }}
QMenuBar {{ background: {BG_APP}; color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background: {SELECTION}; color: {ACCENT}; }}
QMenu {{ background: {BG_PANEL}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 5px 24px 5px 12px; }}
QMenu::item:selected {{ background: {SELECTION}; color: {ACCENT}; }}
QPlainTextEdit {{ background: {BG_PANEL}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; border-radius: 4px;
    selection-background-color: {SELECTION};
    font-family: Consolas, 'DejaVu Sans Mono', Menlo, monospace; }}
QStatusBar {{ background: {BG_RECESS}; color: {TEXT_MUTED};
    border-top: 1px solid {BORDER}; }}
QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; background: {BG_RECESS};
    border: 1px solid {TEXT_MUTED}; border-radius: 3px; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; background: {BG_PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QAbstractSpinBox {{ background: {BG_CTRL}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 4px; }}
QAbstractSpinBox:focus {{ border-color: {ACCENT}; }}
QDialog {{ background: {BG_APP}; }}
QFrame[glass="true"] {{ background: {GLASS_PANEL};
    border: 1px solid {GLASS_BORDER}; border-radius: 6px; }}
QPushButton[destructive="true"] {{ color: {ENGAGED_ORANGE};
    border-color: rgb(90,58,26); }}
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_theme.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/theme.py tests/app/test_theme.py
git commit -m "feat(app): exact Lyra theme tokens + app QSS"
```

---

### Task 7: Persisted settings (`config.py`)

**Files:**
- Create: `deepfistapp/config.py`
- Test: `tests/app/test_config.py`

**Interfaces:**
- Produces (used by Tasks 16, 17): `class AppConfig` wrapping `QSettings`, constructor `AppConfig(settings: QSettings | None = None)` (defaults to `QSettings("DeepFist", "DeepFistTheApp")`; tests inject an ini-backed instance). Typed properties with spec defaults — `device_id: str | None (None)`, `follow_default: bool (True)`, `auto_resume: bool (True)`, `squelch_thresh: float (12.0)`, `wf_floor_db: float (-90.0)`, `wf_gain_db: float (30.0)`, `spec_avg: float (0.6)`, `wf_speed_rows: int (20)`, `wpm_min: int (10)`, `wpm_max: int (40)` (both clamped to hard range 5–60), `model_path: str | None (None)`, `tick_s: float (0.4)`, `guard_s: float (1.3)`, `log_to_file: bool (False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_config.py`:

```python
from PySide6.QtCore import QSettings


def _cfg(tmp_path):
    from deepfistapp.config import AppConfig
    qs = QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat)
    return AppConfig(qs)


def test_defaults_match_spec(tmp_path):
    c = _cfg(tmp_path)
    assert c.device_id is None and c.follow_default is True and c.auto_resume is True
    assert c.squelch_thresh == 12.0
    assert (c.wpm_min, c.wpm_max) == (10, 40)
    assert c.model_path is None
    assert (c.tick_s, c.guard_s) == (0.4, 1.3)
    assert c.log_to_file is False


def test_round_trip_persists(tmp_path):
    c = _cfg(tmp_path)
    c.squelch_thresh = 20.5
    c.device_id = "spk-1"
    c.follow_default = False
    c.wpm_max = 55
    c.model_path = "C:/models/x.onnx"
    c2 = _cfg(tmp_path)
    assert c2.squelch_thresh == 20.5 and c2.device_id == "spk-1"
    assert c2.follow_default is False and c2.wpm_max == 55
    assert c2.model_path == "C:/models/x.onnx"


def test_wpm_bounds_clamped_to_hard_range_5_60(tmp_path):
    c = _cfg(tmp_path)
    c.wpm_min = 1
    c.wpm_max = 99
    assert c.wpm_min == 5 and c.wpm_max == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.config'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/config.py`:

```python
"""QSettings-backed app configuration (org DeepFist, app DeepFistTheApp)."""
from PySide6.QtCore import QSettings

from deepfistapp import APP_NAME, ORG_NAME


def _prop(key, default, typ):
    def get(self):
        v = self._qs.value(key, default)
        if v is None or v == "":
            return default
        if typ is bool:
            return v in (True, "true", "True", 1, "1")
        return typ(v)

    def set_(self, value):
        self._qs.setValue(key, value)
        self._qs.sync()

    return property(get, set_)


def _clamped_int(key, default, lo, hi):
    def get(self):
        v = self._qs.value(key, default)
        return min(hi, max(lo, int(default if v in (None, "") else v)))

    def set_(self, value):
        self._qs.setValue(key, min(hi, max(lo, int(value))))
        self._qs.sync()

    return property(get, set_)


def _opt_str(key):
    def get(self):
        v = self._qs.value(key, None)
        return None if v in (None, "") else str(v)

    def set_(self, value):
        self._qs.setValue(key, "" if value is None else str(value))
        self._qs.sync()

    return property(get, set_)


class AppConfig:
    def __init__(self, settings: QSettings | None = None):
        self._qs = settings or QSettings(ORG_NAME, APP_NAME)

    device_id = _opt_str("routing/device_id")
    follow_default = _prop("routing/follow_default", True, bool)
    auto_resume = _prop("routing/auto_resume", True, bool)
    squelch_thresh = _prop("squelch/thresh", 12.0, float)
    wf_floor_db = _prop("display/wf_floor_db", -90.0, float)
    wf_gain_db = _prop("display/wf_gain_db", 30.0, float)
    spec_avg = _prop("display/spec_avg", 0.6, float)
    wf_speed_rows = _prop("display/wf_speed_rows", 20, int)
    wpm_min = _clamped_int("speed/wpm_min", 10, 5, 60)
    wpm_max = _clamped_int("speed/wpm_max", 40, 5, 60)
    model_path = _opt_str("model/path")
    tick_s = _prop("diag/tick_s", 0.4, float)
    guard_s = _prop("diag/guard_s", 1.3, float)
    log_to_file = _prop("diag/log_to_file", False, bool)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_config.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/config.py tests/app/test_config.py
git commit -m "feat(app): QSettings-backed AppConfig with spec defaults"
```

---

### Task 8: AudioSource protocol + validating WAV source

**Files:**
- Create: `deepfistapp/audio/source.py`, `deepfistapp/audio/wav_source.py`
- Test: `tests/app/test_wav_source.py`

**Interfaces:**
- Produces (used by Tasks 9, 13, 16, 17):
  - `source.py`: `class SourceError(RuntimeError)`; `class AudioSource(Protocol)` with attributes `samplerate: int`, `channels: int` and methods `start() -> None`, `stop() -> None`, `read(timeout: float) -> np.ndarray | None` (`[frames, channels]` float32 in −1..1; `None` on EOF/timeout).
  - `wav_source.py`: `class WavError(SourceError)` (message = user-facing reason); `class WavSource` implementing AudioSource with `__init__(path: str | Path, realtime: bool = True, block_ms: int = 100)`, plus `duration_s: float`, `position_s: float`, `paused: bool` (while True, `read` sleeps briefly and returns an empty `(0, channels)` array — the worker skips empty blocks), `seek(seconds: float) -> None`. `realtime=False` reads without sleeping (tests/integration).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_wav_source.py`:

```python
import struct
import wave

import numpy as np
import pytest


def _write_pcm16(path, sr=8000, seconds=1.0, ch=1):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.5 * np.sin(2 * np.pi * 600 * t) * 32767).astype("<i2")
    data = np.repeat(x[:, None], ch, axis=1).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data)
    return path


def _write_float32(path, sr=8000, seconds=0.5):
    n = int(sr * seconds)
    x = (0.5 * np.sin(2 * np.pi * 600 * np.arange(n) / sr)).astype("<f4")
    data = x.tobytes()
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
           + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32)
           + b"data" + struct.pack("<I", len(data)))
    path.write_bytes(hdr + data)
    return path


def test_reads_pcm16_stereo(tmp_path):
    from deepfistapp.audio.wav_source import WavSource
    p = _write_pcm16(tmp_path / "a.wav", ch=2)
    src = WavSource(p, realtime=False)
    assert src.channels == 2 and src.samplerate == 8000
    assert abs(src.duration_s - 1.0) < 0.01
    src.start()
    block = src.read(timeout=1.0)
    assert block.dtype == np.float32 and block.shape[1] == 2
    assert np.abs(block).max() <= 1.0
    src.stop()


def test_reads_ieee_float32(tmp_path):
    from deepfistapp.audio.wav_source import WavSource
    src = WavSource(_write_float32(tmp_path / "f.wav"), realtime=False)
    src.start()
    assert src.read(timeout=1.0).dtype == np.float32
    src.stop()


def test_eof_returns_none_and_seek_rewinds(tmp_path):
    from deepfistapp.audio.wav_source import WavSource
    src = WavSource(_write_pcm16(tmp_path / "a.wav"), realtime=False)
    src.start()
    while src.read(timeout=1.0) is not None:
        pass
    assert src.read(timeout=1.0) is None
    src.seek(0.0)
    assert src.read(timeout=1.0) is not None
    src.stop()


@pytest.mark.parametrize("build,reason", [
    (lambda p: p.write_bytes(b"NOTAWAVFILE" * 10), "not a WAV file"),
    (lambda p: p.write_bytes(
        b"RIFF" + struct.pack("<I", 36) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 2, 1, 8000, 8000, 1, 4)
        + b"data" + struct.pack("<I", 0)), "compressed"),
    (lambda p: p.write_bytes(
        b"RIFF" + struct.pack("<I", 36) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data" + struct.pack("<I", 0)), "zero-length"),
])
def test_rejections_name_the_reason(tmp_path, build, reason):
    from deepfistapp.audio.wav_source import WavError, WavSource
    p = tmp_path / "bad.wav"
    build(p)
    with pytest.raises(WavError) as e:
        WavSource(p)
    assert reason in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_wav_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.audio.wav_source'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/audio/source.py`:

```python
"""AudioSource protocol — every capture/file source implements this."""
from typing import Protocol, runtime_checkable

import numpy as np


class SourceError(RuntimeError):
    """Capture/read failure (device lost, stream died)."""


@runtime_checkable
class AudioSource(Protocol):
    samplerate: int
    channels: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout: float) -> np.ndarray | None: ...
```

Create `deepfistapp/audio/wav_source.py`:

```python
"""WAV file source — spec §5.2. Own RIFF parser: stdlib wave rejects IEEE
float, and we must name the exact rejection reason."""
import struct
import time
from pathlib import Path

import numpy as np

from deepfistapp.audio.source import SourceError

MAX_BYTES = 2 * 1024**3
_PCM, _FLOAT = 1, 3


class WavError(SourceError):
    pass


def _parse(path: Path):
    if path.stat().st_size > MAX_BYTES:
        raise WavError(f"{path.name} is over 2 GB — split the recording")
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            raise WavError(f"{path.name} is not a WAV file")
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                raise WavError(f"{path.name}: unreadable header (no data chunk)")
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                body = f.read(size)
                if len(body) < 16:
                    raise WavError(f"{path.name}: unreadable header (short fmt)")
                tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:   # WAVE_FORMAT_EXTENSIBLE
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, ch, sr, bits)
            elif cid == b"data":
                if fmt is None:
                    raise WavError(f"{path.name}: unreadable header (data before fmt)")
                tag, ch, sr, bits = fmt
                ok = (tag == _PCM and bits in (8, 16, 24, 32)) or \
                     (tag == _FLOAT and bits == 32)
                if not ok:
                    raise WavError(
                        f"{path.name} uses compressed audio (format tag {tag}, "
                        f"{bits}-bit) — export as 16-bit PCM")
                if ch < 1 or ch > 8:
                    raise WavError(f"{path.name}: {ch} channels unsupported (1-8)")
                if size == 0:
                    raise WavError(f"{path.name}: zero-length audio data")
                if size > path.stat().st_size - f.tell():
                    raise WavError(f"{path.name}: truncated data chunk")
                return tag, ch, sr, bits, f.tell(), size
            else:
                f.seek(size + (size & 1), 1)


def _to_float32(raw: bytes, tag: int, bits: int) -> np.ndarray:
    if tag == _FLOAT:
        return np.frombuffer(raw, "<f4").astype(np.float32)
    if bits == 8:
        return (np.frombuffer(raw, "u1").astype(np.float32) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if bits == 24:
        b = np.frombuffer(raw, "u1").reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v -= (v & 0x800000) << 1
        return v.astype(np.float32) / 8388608.0
    return np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0


class WavSource:
    def __init__(self, path, realtime: bool = True, block_ms: int = 100):
        self.path = Path(path)
        (self._tag, self.channels, self.samplerate, self._bits,
         self._data_off, self._data_len) = _parse(self.path)
        self._bpf = max(1, (self._bits // 8) * self.channels)   # bytes/frame
        self.duration_s = (self._data_len // self._bpf) / self.samplerate
        self.position_s = 0.0
        self.realtime = realtime
        self.paused = False
        self._block_frames = max(1, int(self.samplerate * block_ms / 1000))
        self._f = None

    def start(self) -> None:
        self._f = open(self.path, "rb")
        self.seek(self.position_s)

    def stop(self) -> None:
        if self._f:
            self._f.close()
            self._f = None

    def seek(self, seconds: float) -> None:
        frame = int(max(0.0, min(seconds, self.duration_s)) * self.samplerate)
        self.position_s = frame / self.samplerate
        if self._f:
            self._f.seek(self._data_off + frame * self._bpf)

    def read(self, timeout: float):
        if self._f is None:
            raise SourceError("WavSource.read before start()")
        if self.paused:
            time.sleep(0.1)
            return np.zeros((0, self.channels), dtype=np.float32)
        end = self._data_off + self._data_len
        n = min(self._block_frames, (end - self._f.tell()) // self._bpf)
        if n <= 0:
            return None
        raw = self._f.read(n * self._bpf)
        self.position_s += n / self.samplerate
        if self.realtime:
            time.sleep(n / self.samplerate)
        return _to_float32(raw, self._tag, self._bits).reshape(-1, self.channels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_wav_source.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/audio/source.py deepfistapp/audio/wav_source.py tests/app/test_wav_source.py
git commit -m "feat(app): AudioSource protocol + validating streaming WAV source"
```

---

### Task 9: Output-device enumeration + loopback capture

**Files:**
- Create: `deepfistapp/audio/devices.py`, `deepfistapp/audio/loopback.py`
- Test: `tests/app/test_devices.py`

**Interfaces:**
- Consumes: `soundcard` (imported lazily inside functions so tests can inject a fake and non-audio machines import cleanly); `SourceError` from Task 8.
- Produces (used by Tasks 16, 17):
  - `devices.py`: `@dataclass DeviceInfo(id: str, name: str, is_default: bool)`; `list_output_devices() -> list[DeviceInfo]`; `default_output_id() -> str | None`
  - `loopback.py`: `class LoopbackSource` implementing AudioSource — `__init__(device_id: str | None = None, samplerate: int = 48000, block_ms: int = 100)`; `device_id=None` = follow system default; `read` raises `SourceError` on capture failure (the DEVICE_LOST trigger).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_devices.py`:

```python
import sys
import types

import numpy as np
import pytest


class _FakeMic:
    def __init__(self, id_, name, fail_after=None):
        self.id, self.name, self.isloopback = id_, name, True
        self._fail_after, self._reads = fail_after, 0

    def recorder(self, samplerate, channels, blocksize):
        mic = self

        class _Rec:
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def record(self, numframes):
                mic._reads += 1
                if mic._fail_after is not None and mic._reads > mic._fail_after:
                    raise RuntimeError("device vanished")
                return np.zeros((numframes, channels), dtype=np.float32)
        return _Rec()


class _FakeSpeaker:
    def __init__(self, id_, name):
        self.id, self.name = id_, name


def _fake_soundcard(monkeypatch, mics, speakers, default):
    fake = types.SimpleNamespace(
        all_microphones=lambda include_loopback=True: mics,
        all_speakers=lambda: speakers,
        default_speaker=lambda: default,
        get_microphone=lambda id, include_loopback=True: next(
            m for m in mics if m.id == id),
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake)
    return fake


def test_list_output_devices_marks_default(monkeypatch):
    spk = [_FakeSpeaker("s1", "Speakers"), _FakeSpeaker("s2", "Headphones")]
    _fake_soundcard(monkeypatch, [], spk, spk[1])
    from deepfistapp.audio.devices import list_output_devices
    infos = list_output_devices()
    assert [(d.id, d.is_default) for d in infos] == [("s1", False), ("s2", True)]


def test_loopback_reads_blocks_and_raises_on_device_loss(monkeypatch):
    mic = _FakeMic("s1", "Speakers", fail_after=2)
    _fake_soundcard(monkeypatch, [mic], [_FakeSpeaker("s1", "Speakers")],
                    _FakeSpeaker("s1", "Speakers"))
    from deepfistapp.audio.loopback import LoopbackSource
    from deepfistapp.audio.source import SourceError
    src = LoopbackSource("s1", samplerate=48000, block_ms=10)
    src.start()
    b = src.read(timeout=1.0)
    assert b.shape == (480, 2) and b.dtype == np.float32
    src.read(timeout=1.0)
    with pytest.raises(SourceError):
        src.read(timeout=1.0)
    src.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_devices.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.audio.devices'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/audio/devices.py`:

```python
"""Output-endpoint enumeration (spec §5.1). soundcard is imported lazily so
tests inject a fake and import time stays clean on machines without audio."""
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    id: str
    name: str
    is_default: bool


def default_output_id() -> str | None:
    import soundcard as sc
    try:
        return str(sc.default_speaker().id)
    except Exception:
        return None


def list_output_devices() -> list[DeviceInfo]:
    import soundcard as sc
    default = default_output_id()
    return [DeviceInfo(str(s.id), s.name, str(s.id) == default)
            for s in sc.all_speakers()]
```

Create `deepfistapp/audio/loopback.py`:

```python
"""System-output loopback: WASAPI loopback (Windows), Pulse/PipeWire monitor
sources (Linux), virtual devices e.g. BlackHole (macOS) — soundcard exposes
all of these as loopback-capable microphones keyed by the speaker's id."""
import numpy as np

from deepfistapp.audio.devices import default_output_id
from deepfistapp.audio.source import SourceError


class LoopbackSource:
    channels = 2

    def __init__(self, device_id: str | None = None,
                 samplerate: int = 48000, block_ms: int = 100):
        self.device_id = device_id
        self.samplerate = samplerate
        self._block = max(1, int(samplerate * block_ms / 1000))
        self._rec = None
        self._ctx = None

    def start(self) -> None:
        import soundcard as sc
        target = self.device_id or default_output_id()
        if target is None:
            raise SourceError("no output device available")
        try:
            mic = sc.get_microphone(id=target, include_loopback=True)
            self._ctx = mic.recorder(samplerate=self.samplerate,
                                     channels=self.channels,
                                     blocksize=self._block)
            self._rec = self._ctx.__enter__()
        except Exception as e:
            raise SourceError(f"cannot open loopback for device {target}: {e}") from e

    def stop(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._rec = self._ctx = None

    def read(self, timeout: float):
        if self._rec is None:
            raise SourceError("LoopbackSource.read before start()")
        try:
            data = self._rec.record(numframes=self._block)
        except Exception as e:
            raise SourceError(f"capture failed: {e}") from e
        return np.asarray(data, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_devices.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/audio/devices.py deepfistapp/audio/loopback.py tests/app/test_devices.py
git commit -m "feat(app): output-device enumeration + loopback capture source"
```

---

### Task 10: ONNX session wrapper, model validation, fallback

**Files:**
- Create: `deepfistapp/engine/session.py`
- Test: `tests/app/test_session.py`

**Interfaces:**
- Consumes: `onnxruntime`; test fixture uses `deepfist.export.to_onnx.export_onnx` + torch (dev-only) to build a valid random-weight model.
- Produces (used by Tasks 11, 12, 13, 17):
  - `class ModelError(ValueError)` — message names the failed check
  - `class OnnxSession`: `__init__(model_path: str | Path)` validating per spec §9; attributes `path: Path`, `name: str`, `meta: dict | None`, `warning: str | None` ("no metadata sidecar — assuming standard preprocessing" when sidecar missing); `run(spec: np.ndarray) -> np.ndarray` (`[1,1,65,T]` float32 → `[T',1,48]`)
  - `bundled_model_path() -> Path` (= `deepfistapp/resources/deepfist.onnx`)
  - `load_with_fallback(preferred: str | Path | None, bundled: str | Path) -> tuple[OnnxSession, str | None]` — `(session, user_facing_warning_or_None)`; never raises if the bundled model is valid.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_session.py`:

```python
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def valid_model(tmp_path_factory):
    """Random-weight CwCtcNet exported through the real export path."""
    from deepfist.export.to_onnx import export_onnx, write_metadata
    from deepfist.model.net import CwCtcNet
    d = tmp_path_factory.mktemp("model")
    net = CwCtcNet(width=1.0)
    net.eval()
    out = str(d / "m.onnx")
    export_onnx(net, out)
    write_metadata(out)
    return d / "m.onnx"


def test_valid_model_loads_and_runs(valid_model):
    from deepfistapp.engine.session import OnnxSession
    s = OnnxSession(valid_model)
    assert s.name == "m" and s.warning is None and s.meta is not None
    out = s.run(np.zeros((1, 1, 65, 401), dtype=np.float32))
    assert out.ndim == 3 and out.shape[1] == 1 and out.shape[2] == 48


def test_missing_file_and_garbage_file(tmp_path):
    from deepfistapp.engine.session import ModelError, OnnxSession
    with pytest.raises(ModelError, match="not found"):
        OnnxSession(tmp_path / "nope.onnx")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelError, match="rejected"):
        OnnxSession(bad)


def test_missing_sidecar_warns_mismatched_sidecar_rejects(valid_model, tmp_path):
    from deepfistapp.engine.session import ModelError, OnnxSession
    solo = tmp_path / "solo.onnx"
    shutil.copy(valid_model, solo)
    s = OnnxSession(solo)                    # no sidecar
    assert "no metadata" in s.warning
    meta = json.loads((valid_model.parent / "m.onnx.json").read_text())
    meta["preprocessing"]["sample_rate"] = 8000
    (tmp_path / "solo.onnx.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="sample_rate"):
        OnnxSession(solo)


def test_fallback_uses_bundled_and_reports(valid_model, tmp_path):
    from deepfistapp.engine.session import load_with_fallback
    s, warn = load_with_fallback(tmp_path / "missing.onnx", valid_model)
    assert s.path == valid_model and "using bundled model" in warn
    s2, warn2 = load_with_fallback(None, valid_model)
    assert s2.path == valid_model and warn2 is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.session'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/session.py`:

```python
"""ONNX Runtime session + model validation/fallback (spec §9)."""
import json
from pathlib import Path

import numpy as np

N_CLASSES = 48
_EXPECTED_PRE = {"sample_rate": 3200, "n_fft": 256, "hop_length": 48}


class ModelError(ValueError):
    pass


def bundled_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / "deepfist.onnx"


class OnnxSession:
    def __init__(self, model_path):
        import onnxruntime as ort
        p = Path(model_path)
        if not p.exists():
            raise ModelError(f"model file not found: {p}")
        try:
            self._sess = ort.InferenceSession(
                str(p), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelError(f"ONNX Runtime rejected the file: {e}") from e
        ins = self._sess.get_inputs()
        if len(ins) != 1 or ins[0].name != "spectrogram":
            raise ModelError("input tensor must be named 'spectrogram'")
        shape = ins[0].shape
        if len(shape) != 4 or shape[1] != 1 or shape[2] != 65:
            raise ModelError(f"input shape must be [batch,1,65,time], got {shape}")
        outs = self._sess.get_outputs()
        if len(outs) != 1 or outs[0].name != "log_probs":
            raise ModelError("output tensor must be named 'log_probs'")
        if outs[0].shape[-1] != N_CLASSES:
            raise ModelError(
                f"class dimension must be {N_CLASSES}, got {outs[0].shape[-1]}")
        self.path, self.name = p, p.stem
        sidecar = p.parent / (p.name + ".json")
        if sidecar.exists():
            self.meta = json.loads(sidecar.read_text())
            self.warning = None
            pre = self.meta.get("preprocessing", {})
            for k, v in _EXPECTED_PRE.items():
                if pre.get(k) != v:
                    raise ModelError(
                        f"metadata mismatch: {k}={pre.get(k)} (expected {v})")
        else:
            self.meta = None
            self.warning = "no metadata sidecar — assuming standard preprocessing"

    def run(self, spec: np.ndarray) -> np.ndarray:
        return self._sess.run(
            ["log_probs"], {"spectrogram": spec.astype(np.float32)})[0]


def load_with_fallback(preferred, bundled):
    if preferred is not None:
        try:
            return OnnxSession(preferred), None
        except ModelError as e:
            return OnnxSession(bundled), \
                f"custom model rejected ({e}) — using bundled model"
    return OnnxSession(bundled), None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_session.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/session.py tests/app/test_session.py
git commit -m "feat(app): validated ONNX session with bundled-model fallback"
```

---

### Task 11: Fixed-order pipeline (`engine/pipeline.py`)

**Files:**
- Create: `deepfistapp/engine/pipeline.py`
- Test: `tests/app/test_pipeline.py`

**Interfaces:**
- Consumes: `deepfist.dsp.squelch.has_signal`, `deepfist.dsp.despike.despike`, `deepfist.features.conditioner.condition` / `detect_tone` (numpy-only module), `audio_to_spectrogram_np` (Task 3), `greedy_frames` (Task 4), `OnnxSession` (Task 10).
- Produces (used by Tasks 13, 17):
  - `@dataclass PipelineConfig(squelch_thresh: float = 12.0, use_despike: bool = True, tone_hz: float | None = None, blank_pen: float = 0.0)`
  - `@dataclass WindowResult(active: bool, score: float, ids: list[int], frames: list[int], n_frames: int, tone_hz: float)`
  - `process_window(audio: np.ndarray, session: OnnxSession, cfg: PipelineConfig) -> WindowResult` — `audio` is mono 3200 Hz.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_pipeline.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from deepfist.export.to_onnx import export_onnx, write_metadata
    from deepfist.model.net import CwCtcNet
    from deepfistapp.engine.session import OnnxSession
    d = tmp_path_factory.mktemp("m")
    net = CwCtcNet(width=1.0); net.eval()
    export_onnx(net, str(d / "m.onnx")); write_metadata(str(d / "m.onnx"))
    return OnnxSession(d / "m.onnx")


def _keyed_tone(seconds=6.0, sr=3200, pitch=700.0):
    t = np.arange(int(seconds * sr)) / sr
    gate = (np.sin(2 * np.pi * 4 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * pitch * t) * gate).astype(np.float32)


def test_closed_gate_skips_everything_and_emits_nothing(session, monkeypatch):
    from deepfistapp.engine import pipeline
    calls = []
    monkeypatch.setattr(pipeline, "despike",
                        lambda x, sr: calls.append("despike") or x)
    monkeypatch.setattr(pipeline, "condition",
                        lambda x, sr, tone_hz=None: calls.append("condition") or x)
    silence = np.zeros(6 * 3200, dtype=np.float32)
    res = pipeline.process_window(silence, session,
                                  pipeline.PipelineConfig())
    assert res.active is False and res.ids == [] and calls == []


def test_open_gate_runs_stages_in_fixed_order(session, monkeypatch):
    from deepfistapp.engine import pipeline
    order = []
    real_hs, real_ds = pipeline.has_signal, pipeline.despike
    real_cond, real_spec = pipeline.condition, pipeline.audio_to_spectrogram_np
    monkeypatch.setattr(pipeline, "has_signal",
                        lambda a, sr, th: order.append("squelch") or real_hs(a, sr, th))
    monkeypatch.setattr(pipeline, "despike",
                        lambda a, sr: order.append("despike") or real_ds(a, sr))
    monkeypatch.setattr(pipeline, "condition",
                        lambda a, sr, tone_hz=None: order.append("condition")
                        or real_cond(a, sr, tone_hz=tone_hz))
    monkeypatch.setattr(pipeline, "audio_to_spectrogram_np",
                        lambda a, sr: order.append("spectrogram") or real_spec(a, sr))
    res = pipeline.process_window(_keyed_tone(), session, pipeline.PipelineConfig())
    assert res.active is True
    assert order == ["squelch", "despike", "condition", "spectrogram"]
    assert res.n_frames > 0


def test_manual_tone_override_reported(session):
    from deepfistapp.engine.pipeline import PipelineConfig, process_window
    res = process_window(_keyed_tone(pitch=650.0), session,
                         PipelineConfig(tone_hz=650.0))
    assert res.tone_hz == 650.0


def test_auto_tone_detected_near_pitch(session):
    from deepfistapp.engine.pipeline import PipelineConfig, process_window
    res = process_window(_keyed_tone(pitch=700.0), session, PipelineConfig())
    assert abs(res.tone_hz - 700.0) < 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/pipeline.py`:

```python
"""Fixed-order window processor (global constraint):
raw-audio squelch -> despike -> condition -> spectrogram -> ONNX.
A closed gate short-circuits the whole chain (no-hallucination rule)."""
from dataclasses import dataclass, field

import numpy as np

from deepfist.dsp.despike import despike
from deepfist.dsp.squelch import has_signal
from deepfist.features.conditioner import condition, detect_tone
from deepfistapp.engine.commit import greedy_frames
from deepfistapp.engine.numpy_features import SAMPLE_RATE, audio_to_spectrogram_np


@dataclass
class PipelineConfig:
    squelch_thresh: float = 12.0
    use_despike: bool = True
    tone_hz: float | None = None
    blank_pen: float = 0.0


@dataclass
class WindowResult:
    active: bool
    score: float
    ids: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    n_frames: int = 0
    tone_hz: float = 0.0


def process_window(audio: np.ndarray, session, cfg: PipelineConfig) -> WindowResult:
    active, score = has_signal(audio, SAMPLE_RATE, cfg.squelch_thresh)
    if not active:
        return WindowResult(active=False, score=score)
    clean = despike(audio, SAMPLE_RATE) if cfg.use_despike else audio
    cond = condition(clean, SAMPLE_RATE, tone_hz=cfg.tone_hz)
    spec = audio_to_spectrogram_np(cond, SAMPLE_RATE)[None, None]   # [1,1,65,T]
    log_probs = session.run(spec)                                    # [T',1,48]
    ids, frames, n = greedy_frames(log_probs, cfg.blank_pen)
    tone = cfg.tone_hz if cfg.tone_hz is not None \
        else detect_tone(audio, SAMPLE_RATE)
    return WindowResult(True, score, ids, frames, n, float(tone))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_pipeline.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/pipeline.py tests/app/test_pipeline.py
git commit -m "feat(app): fixed-order decode pipeline with gate short-circuit"
```

---

### Task 12: Display FFT tap (`engine/spectra.py`)

**Files:**
- Create: `deepfistapp/engine/spectra.py`
- Test: `tests/app/test_spectra.py`

**Interfaces:**
- Produces (used by Tasks 13, 14): `N_FFT_DISP = 512`; `N_BINS = 257`; `fft_row(mono_3200: np.ndarray) -> np.ndarray` — float32 `[257]` dB magnitudes of the last 512 samples (0–1600 Hz); `freq_of_bin(i: int) -> float`; `bin_of_freq(hz: float) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_spectra.py`:

```python
import numpy as np


def test_row_shape_and_peak_at_tone():
    from deepfistapp.engine.spectra import N_BINS, bin_of_freq, fft_row, freq_of_bin
    sr = 3200
    t = np.arange(sr) / sr
    row = fft_row(np.sin(2 * np.pi * 700 * t).astype(np.float32))
    assert row.shape == (N_BINS,) and row.dtype == np.float32
    peak = int(np.argmax(row))
    assert abs(freq_of_bin(peak) - 700.0) < 10.0
    assert abs(freq_of_bin(bin_of_freq(700.0)) - 700.0) < 3.2


def test_short_input_is_padded_not_crashed():
    from deepfistapp.engine.spectra import N_BINS, fft_row
    assert fft_row(np.zeros(100, dtype=np.float32)).shape == (N_BINS,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_spectra.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.spectra'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/spectra.py`:

```python
"""Cheap display FFT for the spectrum trace + waterfall rows (spec §4).
512-point at 3200 Hz -> 6.25 Hz/bin over 0-1600 Hz."""
import numpy as np

SAMPLE_RATE = 3200
N_FFT_DISP = 512
N_BINS = N_FFT_DISP // 2 + 1
_HZ_PER_BIN = SAMPLE_RATE / N_FFT_DISP
_WIN = np.hanning(N_FFT_DISP).astype(np.float32)


def fft_row(mono_3200: np.ndarray) -> np.ndarray:
    x = np.asarray(mono_3200, dtype=np.float32)[-N_FFT_DISP:]
    if len(x) < N_FFT_DISP:
        x = np.pad(x, (N_FFT_DISP - len(x), 0))
    mag = np.abs(np.fft.rfft(x * _WIN))
    return (20.0 * np.log10(mag + 1e-9)).astype(np.float32)


def freq_of_bin(i: int) -> float:
    return i * _HZ_PER_BIN


def bin_of_freq(hz: float) -> int:
    return int(round(hz / _HZ_PER_BIN))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_spectra.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/spectra.py tests/app/test_spectra.py
git commit -m "feat(app): display FFT tap with freq/bin mapping"
```

---

### Task 13: Decoder worker thread (`engine/worker.py`)

**Files:**
- Create: `deepfistapp/engine/worker.py`
- Test: `tests/app/test_worker.py`

**Interfaces:**
- Consumes: `AudioSource`/`SourceError` (Task 8), `StateMachine`/`AppState` (Task 5), `PipelineConfig`/`process_window` (Task 11), `commit_new`/`render_text` (Task 4), `fft_row` (Task 12), `deepfist.dsp.tempo.estimate_wpm` (Task 1), `scipy.signal.resample_poly`.
- Produces (used by Tasks 17, 18):
  - `class DecoderWorker(QThread)` with class attr `QUEUE_MAX = 32` and Qt signals `text_committed(str)`, `stats_updated(dict)`, `spectra_row(object)`, `state_changed(object)` (an `AppState`)
  - `__init__(session, cfg: PipelineConfig, tick_s: float = 0.4, guard_s: float = 1.3, window_s: float = 6.0, wpm_bounds: tuple[int, int] = (10, 40))`
  - `attach_source(source: AudioSource, mode: AppState) -> None` (mode is `LISTENING` or `PLAYING`), `request_stop() -> None`, `set_tone(hz: float | None)`, `set_squelch(thresh: float)`, `set_model(session) -> None`, `submit_block(block: np.ndarray) -> None` (thread-safe; public for tests), `overruns: int`
  - stats dict keys: `wpm, score, active, tone_hz, model, overruns, queue_depth, tick_ms, l_db, r_db`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_worker.py`:

```python
import numpy as np
import pytest

from deepfist.morse.alphabet import TOKENS


class FakeSession:
    """Deterministic 'model': one token E anchored to the burst's position.

    The commit rule emits by ABSOLUTE audio time, so the fake must behave like
    a real model: the char stays with the audio event as the window slides.
    It finds the burst by column energy in the spectrogram it is given."""
    name = "fake"

    def run(self, spec):
        n = spec.shape[-1] // 2                       # mimic 2x time downsample
        lp = np.full((n, 1, 48), -10.0, dtype=np.float32)
        lp[:, 0, 0] = -0.1
        energy = spec[0, 0].sum(axis=0)
        if float(energy.max() - energy.min()) > 1.0:  # a burst is in-window
            fr = min(n - 1, int(np.argmax(energy)) // 2)
            lp[fr, 0, 0] = -10.0
            lp[fr, 0, TOKENS.index("E")] = -0.1
        return lp


class FakeSource:
    """Mono 3200 Hz source: paced blocks then EOF (None). Pacing (5 ms/block)
    keeps the bounded queue from overflowing during the test."""
    samplerate, channels = 3200, 1

    def __init__(self, blocks, fail_at=None):
        self._blocks, self._i, self._fail_at = blocks, 0, fail_at

    def start(self): pass
    def stop(self): pass

    def read(self, timeout):
        import time
        from deepfistapp.audio.source import SourceError
        if self._fail_at is not None and self._i >= self._fail_at:
            raise SourceError("gone")
        if self._i >= len(self._blocks):
            return None
        time.sleep(0.005)
        b = self._blocks[self._i]
        self._i += 1
        return b.reshape(-1, 1)


def _burst_blocks(seconds=8.0, sr=3200, block=320, start=2.0, dur=0.5):
    """Silence with one keyed 700 Hz burst at a FIXED absolute time."""
    t = np.arange(int(seconds * sr)) / sr
    gate = ((t >= start) & (t < start + dur)
            & (np.sin(2 * np.pi * 10 * t) > 0)).astype(np.float32)
    x = (np.sin(2 * np.pi * 700 * t) * gate).astype(np.float32)
    return [x[i:i + block] for i in range(0, len(x), block)]


def _run_worker(qtbot, source, seconds=6.0):
    from deepfistapp.engine.pipeline import PipelineConfig
    from deepfistapp.engine.states import AppState
    from deepfistapp.engine.worker import DecoderWorker
    w = DecoderWorker(FakeSession(), PipelineConfig(), tick_s=0.05)
    w.attach_source(source, AppState.PLAYING)
    texts, states = [], []
    w.text_committed.connect(texts.append)
    w.state_changed.connect(states.append)
    with qtbot.waitSignal(w.finished, timeout=int(seconds * 1000) + 8000):
        w.start()
    return w, texts, states


def test_burst_audio_emits_each_char_exactly_once(qtbot):
    w, texts, states = _run_worker(qtbot, FakeSource(_burst_blocks()))
    joined = "".join(texts)
    assert joined.count("E") == 1          # settled-commit: no duplicates
    from deepfistapp.engine.states import AppState
    assert states[-1] is AppState.IDLE     # EOF -> IDLE


def test_silence_emits_no_text(qtbot):
    blocks = [np.zeros(320, dtype=np.float32) for _ in range(60)]
    w, texts, _ = _run_worker(qtbot, FakeSource(blocks))
    assert texts == []


def test_source_error_transitions_to_device_lost(qtbot):
    from deepfistapp.engine.states import AppState
    from deepfistapp.engine.worker import DecoderWorker
    from deepfistapp.engine.pipeline import PipelineConfig
    src = FakeSource(_burst_blocks(seconds=2.0), fail_at=3)
    w = DecoderWorker(FakeSession(), PipelineConfig(), tick_s=0.05)
    w.attach_source(src, AppState.LISTENING)
    states = []
    w.state_changed.connect(states.append)
    with qtbot.waitSignal(w.finished, timeout=10000):
        w.start()
    assert states[-1] is AppState.DEVICE_LOST


def test_queue_bound_drops_oldest_and_counts(qtbot):
    from deepfistapp.engine.pipeline import PipelineConfig
    from deepfistapp.engine.worker import DecoderWorker
    w = DecoderWorker(FakeSession(), PipelineConfig())
    block = np.zeros((320, 1), dtype=np.float32)
    for _ in range(40):
        w.submit_block(block)
    assert w.overruns == 40 - w.QUEUE_MAX == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_worker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.worker'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/worker.py`:

```python
"""Decoder worker (spec §4): capture thread -> bounded queue -> per-tick
window decode -> queued Qt signals. Owns ring buffer + commit boundary."""
import collections
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal
from scipy.signal import resample_poly

from deepfist.dsp.tempo import estimate_wpm
from deepfistapp.audio.source import SourceError
from deepfistapp.engine.commit import commit_new, render_text
from deepfistapp.engine.numpy_features import SAMPLE_RATE
from deepfistapp.engine.pipeline import process_window
from deepfistapp.engine.spectra import fft_row
from deepfistapp.engine.states import AppState, StateMachine


class DecoderWorker(QThread):
    text_committed = Signal(str)
    stats_updated = Signal(dict)
    spectra_row = Signal(object)
    state_changed = Signal(object)

    QUEUE_MAX = 32

    def __init__(self, session, cfg, tick_s=0.4, guard_s=1.3, window_s=6.0,
                 wpm_bounds=(10, 40), parent=None):
        super().__init__(parent)
        self._session, self.cfg = session, cfg
        self.tick_s, self.guard_s, self.window_s = tick_s, guard_s, window_s
        self.wpm_bounds = wpm_bounds
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._sm = StateMachine()
        self._source = None
        self._mode = AppState.LISTENING
        self.overruns = 0
        self._discont = False
        self._capture_error = False
        self._eof = False
        self._wpm = None
        self._ring = None
        self._total = 0
        self._committed_t = 0.0

    # -- control (called from UI thread) --------------------------------
    def attach_source(self, source, mode):
        self._source, self._mode = source, mode

    def request_stop(self):
        self._stop_evt.set()

    def set_paused(self, paused: bool):
        target = AppState.PAUSED if paused else AppState.PLAYING
        if self._sm.state in (AppState.PLAYING, AppState.PAUSED) \
                and self._sm.state is not target:
            self._sm.to(target)
            self.state_changed.emit(self._sm.state)

    def set_tone(self, hz):
        self.cfg.tone_hz = hz

    def set_squelch(self, thresh):
        self.cfg.squelch_thresh = float(thresh)

    def set_model(self, session):
        self._session = session
        self._reset()

    # -- capture side ---------------------------------------------------
    def submit_block(self, block):
        with self._lock:
            if len(self._queue) >= self.QUEUE_MAX:
                self._queue.popleft()
                self.overruns += 1
                self._discont = True
            self._queue.append(np.asarray(block, dtype=np.float32))

    def _capture_loop(self):
        while not self._stop_evt.is_set():
            try:
                block = self._source.read(timeout=0.5)
            except SourceError:
                self._capture_error = True
                return
            if block is None:
                self._eof = True
                return
            if block.size == 0:            # paused source — nothing to queue
                continue
            self.submit_block(block)

    def _drain(self):
        with self._lock:
            blocks = list(self._queue)
            self._queue.clear()
        return blocks

    def _reset(self):
        with self._lock:
            self._queue.clear()
        if self._ring is not None:
            self._ring[:] = 0.0
        self._total = 0
        self._committed_t = 0.0
        self._discont = False

    # -- decode loop ----------------------------------------------------
    def run(self):
        src = self._source
        self._ring = None
        self._total = 0
        self._committed_t = 0.0
        try:
            src.start()
        except SourceError:
            self._sm.state = AppState.LISTENING       # entered via UI intent
            self._sm.to(AppState.DEVICE_LOST)
            self.state_changed.emit(self._sm.state)
            return
        self._sm.to(self._mode)
        self.state_changed.emit(self._sm.state)
        cap = threading.Thread(target=self._capture_loop, daemon=True)
        cap.start()
        sr = src.samplerate
        self._ring = np.zeros(int(self.window_s * sr), dtype=np.float32)
        l_db = r_db = -60.0
        while not self._stop_evt.is_set():
            time.sleep(self.tick_s)
            t0 = time.perf_counter()
            for block in self._drain():
                mono = block.mean(axis=1) if block.ndim > 1 else block
                n = len(mono)
                self._total += n
                if n >= len(self._ring):
                    self._ring[:] = mono[-len(self._ring):]
                else:
                    self._ring[:-n] = self._ring[n:]
                    self._ring[-n:] = mono
                if block.ndim > 1 and block.shape[1] >= 2:
                    l_db = 20 * np.log10(np.sqrt((block[:, 0]**2).mean()) + 1e-9)
                    r_db = 20 * np.log10(np.sqrt((block[:, 1]**2).mean()) + 1e-9)
                else:
                    l_db = r_db = 20 * np.log10(np.sqrt((mono**2).mean()) + 1e-9)
            if self._capture_error:
                self._sm.to(AppState.DEVICE_LOST)
                self.state_changed.emit(self._sm.state)
                break
            if self._sm.state is AppState.PAUSED:
                continue
            final = self._eof and not self._queue   # decide BEFORE decoding;
            if self._total == 0:                    # decode the tail, THEN exit
                if final:
                    self._sm.to(AppState.IDLE)
                    self.state_changed.emit(self._sm.state)
                    break
                continue
            audio_end = self._total / sr
            audio = self._ring if sr == SAMPLE_RATE else \
                resample_poly(self._ring, SAMPLE_RATE, sr).astype(np.float32)
            self.spectra_row.emit(fft_row(audio))
            if self._discont:
                self._committed_t = max(self._committed_t,
                                        audio_end - self.guard_s)
                self._discont = False
            res = process_window(audio, self._session, self.cfg)
            if not res.active:
                self._committed_t = max(self._committed_t,
                                        audio_end - self.guard_s)
            else:
                emit_ids, self._committed_t = commit_new(
                    res.ids, res.frames, res.n_frames,
                    window_s=self.window_s, audio_end_s=audio_end,
                    committed_t=self._committed_t, guard_s=self.guard_s)
                text = render_text(emit_ids)
                if text:
                    self.text_committed.emit(text)
                raw_wpm = estimate_wpm(audio, SAMPLE_RATE)
                lo, hi = self.wpm_bounds
                if raw_wpm and lo <= raw_wpm <= hi:
                    self._wpm = raw_wpm if self._wpm is None \
                        else 0.7 * self._wpm + 0.3 * raw_wpm
            self.stats_updated.emit({
                "wpm": self._wpm if res.active else None,
                "score": res.score, "active": res.active,
                "tone_hz": res.tone_hz if res.active else None,
                "model": getattr(self._session, "name", "?"),
                "overruns": self.overruns,
                "queue_depth": len(self._queue),
                "tick_ms": (time.perf_counter() - t0) * 1000.0,
                "l_db": float(l_db), "r_db": float(r_db),
            })
            if final:
                self._sm.to(AppState.IDLE)
                self.state_changed.emit(self._sm.state)
                break
        self._stop_evt.set()
        src.stop()
        if self._sm.state in (AppState.LISTENING, AppState.PLAYING,
                              AppState.PAUSED):
            self._sm.to(AppState.IDLE)
            self.state_changed.emit(self._sm.state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_worker.py -q`
Expected: `4 passed` (the burst test streams ~80 blocks at 5 ms pacing — a few seconds of wall time, well under the waitSignal timeout)

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/worker.py tests/app/test_worker.py
git commit -m "feat(app): QThread decoder worker with bounded queue and settled commits"
```

---

### Task 14: Waterfall, spectrum, meter widgets

**Files:**
- Create: `deepfistapp/ui/waterfall.py`, `deepfistapp/ui/spectrum.py`, `deepfistapp/ui/meters.py`
- Test: `tests/app/test_ui_panels.py`

**Interfaces:**
- Consumes: `theme` tokens (Task 6), `spectra.freq_of_bin`/`bin_of_freq`/`N_BINS` (Task 12).
- Produces (used by Task 17):
  - `class WaterfallWidget(QWidget)`: `add_row(row: np.ndarray) -> None` (dB row, length `N_BINS`); `set_marker(hz: float | None, manual: bool) -> None`; `set_levels(floor_db: float, gain_db: float) -> None`; Qt signal `tone_clicked(float)` (emitted with the clicked frequency in Hz on left-click); display span fixed 300–1200 Hz.
  - `class SpectrumWidget(QWidget)`: `set_row(row: np.ndarray) -> None` (EMA-averaged trace; `set_averaging(alpha: float)`); same 300–1200 Hz span; glass-chip painted frame.
  - `class MetersWidget(QWidget)`: `set_levels(l_db: float, r_db: float) -> None` (−60..0 dBFS bars).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_ui_panels.py`:

```python
import numpy as np
from PySide6.QtCore import QPoint, Qt


def _row(peak_hz=700.0):
    from deepfistapp.engine.spectra import N_BINS, bin_of_freq
    r = np.full(N_BINS, -80.0, dtype=np.float32)
    r[bin_of_freq(peak_hz)] = -10.0
    return r


def test_waterfall_click_emits_frequency(qtbot):
    from deepfistapp.ui.waterfall import WaterfallWidget
    w = WaterfallWidget()
    qtbot.addWidget(w)
    w.resize(900, 200)
    for _ in range(5):
        w.add_row(_row())
    clicked = []
    w.tone_clicked.connect(clicked.append)
    # x for 700 Hz on the 300-1200 Hz span: (700-300)/900 of the width
    x = int(w.width() * (700.0 - 300.0) / 900.0)
    qtbot.mouseClick(w, Qt.MouseButton.LeftButton, pos=QPoint(x, 100))
    assert len(clicked) == 1 and abs(clicked[0] - 700.0) < 15.0


def test_waterfall_marker_and_levels_accepted(qtbot):
    from deepfistapp.ui.waterfall import WaterfallWidget
    w = WaterfallWidget()
    qtbot.addWidget(w)
    w.set_marker(650.0, manual=True)
    w.set_marker(None, manual=False)
    w.set_levels(-100.0, 40.0)
    w.add_row(_row())          # must not raise after settings changes


def test_spectrum_averages_rows(qtbot):
    from deepfistapp.ui.spectrum import SpectrumWidget
    s = SpectrumWidget()
    qtbot.addWidget(s)
    s.set_averaging(0.5)
    s.set_row(_row())
    s.set_row(_row(peak_hz=900.0))
    assert s._trace is not None and len(s._trace) > 0


def test_meters_clamp_range(qtbot):
    from deepfistapp.ui.meters import MetersWidget
    m = MetersWidget()
    qtbot.addWidget(m)
    m.set_levels(5.0, -120.0)     # clamps to 0 / -60, repaints without error
    assert m._l == 0.0 and m._r == -60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_panels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.ui.waterfall'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/ui/waterfall.py`:

```python
"""Clickable waterfall (spec §7.1) — QImage rows, 300-1200 Hz span."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from deepfistapp import theme
from deepfistapp.engine.spectra import bin_of_freq

SPAN_LO_HZ, SPAN_HI_HZ = 300.0, 1200.0
_ROWS = 200


class WaterfallWidget(QWidget):
    tone_clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._lo = bin_of_freq(SPAN_LO_HZ)
        self._hi = bin_of_freq(SPAN_HI_HZ) + 1
        self._img = QImage(self._hi - self._lo, _ROWS, QImage.Format.Format_RGB32)
        self._img.fill(QColor(16, 20, 28))
        self._floor, self._gain = -90.0, 30.0
        self._marker_hz, self._marker_manual = None, False

    def set_levels(self, floor_db: float, gain_db: float):
        self._floor, self._gain = float(floor_db), float(gain_db)

    def set_marker(self, hz, manual: bool):
        self._marker_hz, self._marker_manual = hz, manual
        self.update()

    def add_row(self, row: np.ndarray):
        band = row[self._lo:self._hi]
        norm = np.clip((band - self._floor) / max(1.0, self._gain), 0.0, 1.0)
        # cool-CRT ramp: recess blue-black -> electric cyan
        rgb = (np.stack([norm * 0, norm * 229, norm * 255], axis=1)
               .astype(np.uint32))
        pixels = (0xFF000000 | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2])
        # scroll down one row, write new row at top
        buf = self._img.bits()
        w = self._img.width()
        arr = np.frombuffer(buf, dtype=np.uint32).reshape(_ROWS, w)
        arr[1:] = arr[:-1]
        arr[0] = pixels
        self.update()

    def _hz_at_x(self, x: int) -> float:
        frac = min(1.0, max(0.0, x / max(1, self.width())))
        return SPAN_LO_HZ + frac * (SPAN_HI_HZ - SPAN_LO_HZ)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.tone_clicked.emit(self._hz_at_x(int(ev.position().x())))

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.drawImage(self.rect(), self._img)
        if self._marker_hz is not None:
            frac = (self._marker_hz - SPAN_LO_HZ) / (SPAN_HI_HZ - SPAN_LO_HZ)
            x = int(frac * self.width())
            color = theme.DIAL_MARKER if self._marker_manual else theme.RX_MARKER
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(x, 0, x, self.height())
        p.end()
```

Create `deepfistapp/ui/spectrum.py`:

```python
"""Spectrum trace on a glass chip (spec §7.1)."""
import numpy as np
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from deepfistapp import theme
from deepfistapp.engine.spectra import bin_of_freq

SPAN_LO_HZ, SPAN_HI_HZ = 300.0, 1200.0


class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._lo = bin_of_freq(SPAN_LO_HZ)
        self._hi = bin_of_freq(SPAN_HI_HZ) + 1
        self._trace = None
        self._alpha = 0.6

    def set_averaging(self, alpha: float):
        self._alpha = min(0.95, max(0.0, float(alpha)))

    def set_row(self, row: np.ndarray):
        band = np.asarray(row[self._lo:self._hi], dtype=np.float32)
        self._trace = band if self._trace is None else \
            self._alpha * self._trace + (1 - self._alpha) * band
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0x14, 0x20, 0x2a, 0xcc))   # GLASS_PANEL
        p.setPen(QPen(QColor(theme.GLASS_BORDER), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._trace is not None:
            t = np.clip((self._trace + 90.0) / 60.0, 0.0, 1.0)
            n, h, w = len(t), self.height(), self.width()
            p.setPen(QPen(QColor(theme.ACCENT), 1))
            pts = [(int(i * w / n), int(h - 4 - t[i] * (h - 8)))
                   for i in range(n)]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a[0], a[1], b[0], b[1])
        p.end()
```

Create `deepfistapp/ui/meters.py`:

```python
"""L/R peak/RMS meters, -60..0 dBFS (spec §7.1)."""
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from deepfistapp import theme


class MetersWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(120)
        self._l = self._r = -60.0

    def set_levels(self, l_db: float, r_db: float):
        self._l = min(0.0, max(-60.0, float(l_db)))
        self._r = min(0.0, max(-60.0, float(r_db)))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 20, 28))
        w = self.width() - 40
        for i, (label, db) in enumerate((("L", self._l), ("R", self._r))):
            y = 8 + i * 22
            frac = (db + 60.0) / 60.0
            p.setPen(QColor(theme.TEXT_MUTED))
            p.drawText(4, y + 12, label)
            p.fillRect(20, y, w, 14, QColor(10, 13, 18))
            p.fillRect(20, y, int(w * frac), 14, QColor(0, 229, 255))
            p.drawText(24 + w, y + 12, f"{db:.0f}")
        p.end()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_panels.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/ui/waterfall.py deepfistapp/ui/spectrum.py deepfistapp/ui/meters.py tests/app/test_ui_panels.py
git commit -m "feat(app): waterfall/spectrum/meter widgets with click-to-tone"
```

---

### Task 15: Transcript, stats bar, banners, transport strip

**Files:**
- Create: `deepfistapp/ui/transcript.py`, `deepfistapp/ui/stats_bar.py`, `deepfistapp/ui/banners.py`, `deepfistapp/ui/transport.py`
- Test: `tests/app/test_ui_main.py` (part 1 — extended in Task 17)

**Interfaces:**
- Consumes: `theme` (Task 6).
- Produces (used by Task 17):
  - `class TranscriptView(QWidget)`: `append_text(text: str)`, `add_marker(line: str)` (e.g. `— model: exp27_bt —`), `text() -> str`, `copy_all()`, `save_to(path: str)`, `request_clear()` (inline two-click confirm: first click arms the button with `ENGAGED_ORANGE` text "Confirm clear", second click clears), signal `cleared()`.
  - `class StatsBar(QWidget)`: `update_stats(stats: dict)` using the Task 13 stats keys; renders `WPM — · signal 3 (closed) · tone — · model exp27_bt` with em-dashes when gate closed.
  - `class Banner(QWidget)`: `show_message(text: str, persistent: bool = False)`, `clear()`; non-persistent messages auto-hide after 6 s (QTimer).
  - `class TransportStrip(QWidget)`: signals `play_pause()`, `stop()`, `seek(float)`; `set_position(pos_s: float, dur_s: float)`; `set_playing(bool)`; hidden by default (`setVisible(False)` in constructor).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_ui_main.py`:

```python
def test_transcript_append_copy_clear_confirm(qtbot):
    from deepfistapp.ui.transcript import TranscriptView
    tv = TranscriptView()
    qtbot.addWidget(tv)
    tv.append_text("CQ CQ DE WI9FD")
    tv.add_marker("— model: fake —")
    assert "CQ CQ DE WI9FD" in tv.text() and "— model: fake —" in tv.text()
    tv.request_clear()                    # arm
    assert tv.text() != ""                # not yet cleared
    tv.request_clear()                    # confirm
    assert tv.text() == ""


def test_transcript_save(qtbot, tmp_path):
    from deepfistapp.ui.transcript import TranscriptView
    tv = TranscriptView()
    qtbot.addWidget(tv)
    tv.append_text("TEST DE N9BC")
    out = tmp_path / "copy.txt"
    tv.save_to(str(out))
    assert out.read_text(encoding="utf-8") == "TEST DE N9BC"


def test_stats_bar_open_and_closed_gate(qtbot):
    from deepfistapp.ui.stats_bar import StatsBar
    sb = StatsBar()
    qtbot.addWidget(sb)
    sb.update_stats({"wpm": 22.4, "score": 34.0, "active": True,
                     "tone_hz": 612.0, "model": "exp27_bt",
                     "overruns": 0, "queue_depth": 1, "tick_ms": 80.0,
                     "l_db": -18.0, "r_db": -21.0})
    assert "22" in sb.text() and "612" in sb.text() and "exp27_bt" in sb.text()
    sb.update_stats({"wpm": None, "score": 3.1, "active": False,
                     "tone_hz": None, "model": "exp27_bt",
                     "overruns": 0, "queue_depth": 0, "tick_ms": 5.0,
                     "l_db": -60.0, "r_db": -60.0})
    assert "—" in sb.text() and "closed" in sb.text()


def test_banner_and_transport(qtbot):
    from deepfistapp.ui.banners import Banner
    from deepfistapp.ui.transport import TransportStrip
    b = Banner()
    qtbot.addWidget(b)
    b.show_message("device lost", persistent=True)
    assert b.isVisible() is False or b.text() == "device lost"  # offscreen: text set
    t = TransportStrip()
    qtbot.addWidget(t)
    assert t.isVisibleTo(t.parentWidget()) is False or not t.isVisible()
    seeks = []
    t.seek.connect(seeks.append)
    t.set_position(30.0, 120.0)
    t._slider.setValue(500)               # user drag to 50% of 0-1000 range
    t._slider.sliderReleased.emit()
    assert seeks and abs(seeks[0] - 60.0) < 1.5   # 50% of 120 s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_main.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.ui.transcript'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/ui/transcript.py`:

```python
"""Decoded-text area + Copy/Save/Clear (spec §7.1)."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QHBoxLayout, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)


class TranscriptView(QWidget):
    cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edit = QPlainTextEdit(readOnly=True)
        self._btn_copy = QPushButton("Copy")
        self._btn_save = QPushButton("Save…")
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setProperty("destructive", True)
        self._armed = False
        row = QHBoxLayout()
        row.addStretch(1)
        for b in (self._btn_copy, self._btn_save, self._btn_clear):
            row.addWidget(b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._edit, 1)
        lay.addLayout(row)
        self._btn_copy.clicked.connect(self.copy_all)
        self._btn_clear.clicked.connect(self.request_clear)

    def append_text(self, text: str):
        self._disarm()
        cursor = self._edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._edit.setTextCursor(cursor)
        self._edit.ensureCursorVisible()

    def add_marker(self, line: str):
        self.append_text(f"\n{line}\n")

    def text(self) -> str:
        return self._edit.toPlainText()

    def copy_all(self):
        QGuiApplication.clipboard().setText(self.text())

    def save_to(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text())

    def _disarm(self):
        if self._armed:
            self._armed = False
            self._btn_clear.setText("Clear")

    def request_clear(self):
        if not self._armed:
            self._armed = True
            self._btn_clear.setText("Confirm clear")
            return
        self._disarm()
        self._edit.clear()
        self.cleared.emit()
```

Create `deepfistapp/ui/stats_bar.py`:

```python
"""WPM / signal / tone / model statistics row (spec §7.1)."""
from PySide6.QtWidgets import QLabel


class StatsBar(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("WPM — · signal — · tone — · model —")

    def update_stats(self, s: dict):
        wpm = f"{s['wpm']:.0f}" if s.get("wpm") else "—"
        gate = "open" if s.get("active") else "closed"
        tone = f"{s['tone_hz']:.0f} Hz" if s.get("tone_hz") else "—"
        self.setText(f"WPM {wpm} · signal {s.get('score', 0):.0f} ({gate}) · "
                     f"tone {tone} · model {s.get('model', '—')}")
```

Create `deepfistapp/ui/banners.py`:

```python
"""Non-blocking notification strip (spec §3.2 banners)."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel

from deepfistapp import theme


class Banner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {theme.BG_PANEL}; "
                           f"color: {theme.ENGAGED_ORANGE}; "
                           f"border: 1px solid {theme.BORDER}; padding: 4px 8px;")
        self.setVisible(False)
        self._timer = QTimer(self, singleShot=True)
        self._timer.timeout.connect(self.clear)

    def show_message(self, text: str, persistent: bool = False):
        self.setText(text)
        self.setVisible(True)
        if not persistent:
            self._timer.start(6000)

    def clear(self):
        self.setVisible(False)
        self.setText("")
```

Create `deepfistapp/ui/transport.py`:

```python
"""Compact WAV transport strip — visible only in WAV mode (spec §5.2)."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


def _fmt(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


class TransportStrip(QWidget):
    play_pause = Signal()
    stop = Signal()
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self._btn_play = QPushButton("⏸")
        self._btn_stop = QPushButton("⏹")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._label = QLabel("0:00 / 0:00")
        self._dur = 0.0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(self._btn_play)
        lay.addWidget(self._btn_stop)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._label)
        self._btn_play.clicked.connect(self.play_pause.emit)
        self._btn_stop.clicked.connect(self.stop.emit)
        self._slider.sliderReleased.connect(
            lambda: self.seek.emit(self._slider.value() / 1000.0 * self._dur))

    def set_playing(self, playing: bool):
        self._btn_play.setText("⏸" if playing else "▶")

    def set_position(self, pos_s: float, dur_s: float):
        self._dur = dur_s
        if not self._slider.isSliderDown() and dur_s > 0:
            self._slider.setValue(int(pos_s / dur_s * 1000))
        self._label.setText(f"{_fmt(pos_s)} / {_fmt(dur_s)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_main.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/ui/transcript.py deepfistapp/ui/stats_bar.py deepfistapp/ui/banners.py deepfistapp/ui/transport.py tests/app/test_ui_main.py
git commit -m "feat(app): transcript, stats bar, banner, WAV transport widgets"
```

---

### Task 16: Settings drawer

**Files:**
- Create: `deepfistapp/ui/settings_drawer.py`
- Test: `tests/app/test_ui_settings.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 7), `DeviceInfo`/`list_output_devices` (Task 9), `theme` (Task 6).
- Produces (used by Task 17): `class SettingsDrawer(QFrame)` — `__init__(config: AppConfig, devices_fn: Callable[[], list[DeviceInfo]] = list_output_devices)`; sections Routing / Squelch / Display / Speed bounds / Model / Diagnostics; every control live-applies to `config` on change and emits `changed(str, object)` (key path, new value); `refresh_devices()`; `set_live_score(score: float)` (squelch calibration meter); `set_diagnostics(stats: dict)`; `set_model_info(name: str, path: str, is_bundled: bool)`; signals `load_model_requested()`, `revert_model_requested()`, `rescan_requested()`; `toggle()` shows/hides (drawer slide is a fixed-width show/hide; glass backdrop via `setProperty("glass", True)`).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_ui_settings.py`:

```python
from PySide6.QtCore import QSettings


def _cfg(tmp_path):
    from deepfistapp.config import AppConfig
    return AppConfig(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))


def _fake_devices():
    from deepfistapp.audio.devices import DeviceInfo
    return [DeviceInfo("s1", "Speakers", True), DeviceInfo("s2", "Phones", False)]


def test_drawer_builds_and_lists_devices(qtbot, tmp_path):
    from deepfistapp.ui.settings_drawer import SettingsDrawer
    d = SettingsDrawer(_cfg(tmp_path), devices_fn=_fake_devices)
    qtbot.addWidget(d)
    assert d._device_combo.count() == 2
    assert d.property("glass") is True


def test_squelch_slider_live_applies_to_config(qtbot, tmp_path):
    from deepfistapp.ui.settings_drawer import SettingsDrawer
    cfg = _cfg(tmp_path)
    d = SettingsDrawer(cfg, devices_fn=_fake_devices)
    qtbot.addWidget(d)
    changes = []
    d.changed.connect(lambda k, v: changes.append((k, v)))
    d._squelch_slider.setValue(20)
    assert cfg.squelch_thresh == 20.0
    assert ("squelch/thresh", 20.0) in changes


def test_wpm_bounds_persist_and_clamp(qtbot, tmp_path):
    from deepfistapp.ui.settings_drawer import SettingsDrawer
    cfg = _cfg(tmp_path)
    d = SettingsDrawer(cfg, devices_fn=_fake_devices)
    qtbot.addWidget(d)
    d._wpm_min.setValue(8)
    d._wpm_max.setValue(50)
    assert cfg.wpm_min == 8 and cfg.wpm_max == 50


def test_model_signals(qtbot, tmp_path):
    from deepfistapp.ui.settings_drawer import SettingsDrawer
    d = SettingsDrawer(_cfg(tmp_path), devices_fn=_fake_devices)
    qtbot.addWidget(d)
    fired = []
    d.load_model_requested.connect(lambda: fired.append("load"))
    d.revert_model_requested.connect(lambda: fired.append("revert"))
    d._btn_load_model.click()
    d._btn_revert_model.click()
    assert fired == ["load", "revert"]
    d.set_model_info("exp27_bt", "C:/x/deepfist.onnx", is_bundled=True)
    assert "exp27_bt" in d._model_label.text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_settings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.ui.settings_drawer'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/ui/settings_drawer.py`:

```python
"""Right-edge settings drawer (spec §7.3) — live-applies every control."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QLabel,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout)

from deepfistapp.audio.devices import list_output_devices


def _header(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet("color: rgb(0,229,255); font-weight: 800; "
                      "letter-spacing: 1.5px; padding-top: 8px;")
    return lbl


class SettingsDrawer(QFrame):
    changed = Signal(str, object)
    load_model_requested = Signal()
    revert_model_requested = Signal()
    rescan_requested = Signal()

    def __init__(self, config, devices_fn=list_output_devices, parent=None):
        super().__init__(parent)
        self.setProperty("glass", True)
        self.setFixedWidth(320)
        self.setVisible(False)
        self._cfg = config
        self._devices_fn = devices_fn
        lay = QVBoxLayout(self)

        lay.addWidget(_header("Routing"))
        self._device_combo = QComboBox()
        self._follow_default = QCheckBox("Follow system default")
        self._auto_resume = QCheckBox("Auto-resume on reconnect")
        self._btn_rescan = QPushButton("Rescan devices")
        for w in (self._device_combo, self._follow_default,
                  self._auto_resume, self._btn_rescan):
            lay.addWidget(w)

        lay.addWidget(_header("Squelch"))
        self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
        self._squelch_slider.setRange(0, 50)
        self._score_label = QLabel("live score: —")
        lay.addWidget(self._squelch_slider)
        lay.addWidget(self._score_label)

        lay.addWidget(_header("Display"))
        self._wf_floor = QSlider(Qt.Orientation.Horizontal)
        self._wf_floor.setRange(-120, -40)
        self._wf_gain = QSlider(Qt.Orientation.Horizontal)
        self._wf_gain.setRange(10, 60)
        lay.addWidget(QLabel("Waterfall floor"))
        lay.addWidget(self._wf_floor)
        lay.addWidget(QLabel("Waterfall gain"))
        lay.addWidget(self._wf_gain)

        lay.addWidget(_header("Speed bounds"))
        self._wpm_min = QSpinBox(minimum=5, maximum=60)
        self._wpm_max = QSpinBox(minimum=5, maximum=60)
        lay.addWidget(QLabel("Min WPM")); lay.addWidget(self._wpm_min)
        lay.addWidget(QLabel("Max WPM")); lay.addWidget(self._wpm_max)

        lay.addWidget(_header("Model"))
        self._model_label = QLabel("—")
        self._btn_load_model = QPushButton("Load model…")
        self._btn_revert_model = QPushButton("Revert to bundled")
        for w in (self._model_label, self._btn_load_model,
                  self._btn_revert_model):
            lay.addWidget(w)

        lay.addWidget(_header("Diagnostics"))
        self._diag_label = QLabel("queue 0 · overruns 0 · tick — ms")
        self._log_toggle = QCheckBox("Log session to file")
        lay.addWidget(self._diag_label)
        lay.addWidget(self._log_toggle)
        lay.addStretch(1)

        self._load_from_config()
        self._wire()

    # ---- population + wiring -----------------------------------------
    def _load_from_config(self):
        c = self._cfg
        self.refresh_devices()
        self._follow_default.setChecked(c.follow_default)
        self._auto_resume.setChecked(c.auto_resume)
        self._squelch_slider.setValue(int(c.squelch_thresh))
        self._wf_floor.setValue(int(c.wf_floor_db))
        self._wf_gain.setValue(int(c.wf_gain_db))
        self._wpm_min.setValue(c.wpm_min)
        self._wpm_max.setValue(c.wpm_max)
        self._log_toggle.setChecked(c.log_to_file)

    def _apply(self, attr, key, value):
        setattr(self._cfg, attr, value)
        self.changed.emit(key, getattr(self._cfg, attr))

    def _wire(self):
        self._device_combo.currentIndexChanged.connect(
            lambda i: self._apply("device_id", "routing/device_id",
                                  self._device_combo.itemData(i)))
        self._follow_default.toggled.connect(
            lambda v: self._apply("follow_default", "routing/follow_default", v))
        self._auto_resume.toggled.connect(
            lambda v: self._apply("auto_resume", "routing/auto_resume", v))
        self._btn_rescan.clicked.connect(self.rescan_requested.emit)
        self._squelch_slider.valueChanged.connect(
            lambda v: self._apply("squelch_thresh", "squelch/thresh", float(v)))
        self._wf_floor.valueChanged.connect(
            lambda v: self._apply("wf_floor_db", "display/wf_floor_db", float(v)))
        self._wf_gain.valueChanged.connect(
            lambda v: self._apply("wf_gain_db", "display/wf_gain_db", float(v)))
        self._wpm_min.valueChanged.connect(
            lambda v: self._apply("wpm_min", "speed/wpm_min", v))
        self._wpm_max.valueChanged.connect(
            lambda v: self._apply("wpm_max", "speed/wpm_max", v))
        self._btn_load_model.clicked.connect(self.load_model_requested.emit)
        self._btn_revert_model.clicked.connect(self.revert_model_requested.emit)
        self._log_toggle.toggled.connect(
            lambda v: self._apply("log_to_file", "diag/log_to_file", v))

    # ---- runtime updates ---------------------------------------------
    def refresh_devices(self):
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        for d in self._devices_fn():
            label = f"{d.name}{'  (default)' if d.is_default else ''}"
            self._device_combo.addItem(label, d.id)
        self._device_combo.blockSignals(False)

    def set_live_score(self, score: float):
        self._score_label.setText(f"live score: {score:.0f}")

    def set_diagnostics(self, stats: dict):
        self._diag_label.setText(
            f"queue {stats.get('queue_depth', 0)} · "
            f"overruns {stats.get('overruns', 0)} · "
            f"tick {stats.get('tick_ms', 0):.0f} ms")

    def set_model_info(self, name: str, path: str, is_bundled: bool):
        tag = "bundled" if is_bundled else "custom"
        self._model_label.setText(f"{name} ({tag})\n{path}")

    def toggle(self):
        self.setVisible(not self.isVisible())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_settings.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/ui/settings_drawer.py tests/app/test_ui_settings.py
git commit -m "feat(app): live-apply settings drawer"
```

---

### Task 17: Main window + entry point wiring

**Files:**
- Create: `deepfistapp/ui/main_window.py`, `deepfistapp/main.py`
- Test: `tests/app/test_ui_window.py`

**Interfaces:**
- Consumes: everything above — `DecoderWorker` (13), all widgets (14–16), `AppConfig` (7), `theme` (6), `AppState` (5), `OnnxSession`/`load_with_fallback`/`bundled_model_path` (10), `WavSource`/`WavError` (8), `LoopbackSource`/`list_output_devices` (9), `PipelineConfig` (11).
- Produces:
  - `class MainWindow(QMainWindow)`: `__init__(config: AppConfig, session: OnnxSession, startup_warning: str | None = None, devices_fn=list_output_devices, source_factory=None)` — `source_factory(device_id) -> AudioSource` defaults to `LoopbackSource`; tests inject fakes. Public slots/methods: `open_wav(path: str)`, `toggle_listen()`, `set_mode_manual(manual: bool)`, `load_model(path: str)`, `revert_model()`. Layout per spec §7.1: toolbar (title, source combo, Open WAV, status chip, AUTO/MANUAL, ⚙), waterfall above spectrum with meters column spanning both, transcript, stats row, hidden transport, banner strip. Rescan `QTimer` (2000 ms) runs while `DEVICE_LOST`.
  - `deepfistapp/main.py`: `main() -> int` — QApplication, org/app names, `build_qss()`, `AppConfig`, `load_with_fallback(cfg.model_path, bundled_model_path())`, MainWindow, optional `--wav <path>` argument, `app.exec()`.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_ui_window.py`:

```python
import numpy as np
import pytest
from PySide6.QtCore import QSettings

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from deepfist.export.to_onnx import export_onnx, write_metadata
    from deepfist.model.net import CwCtcNet
    from deepfistapp.engine.session import OnnxSession
    d = tmp_path_factory.mktemp("m")
    net = CwCtcNet(width=1.0); net.eval()
    export_onnx(net, str(d / "m.onnx")); write_metadata(str(d / "m.onnx"))
    return OnnxSession(d / "m.onnx")


def _fake_devices():
    from deepfistapp.audio.devices import DeviceInfo
    return [DeviceInfo("s1", "Speakers", True)]


class _IdleSource:
    samplerate, channels = 3200, 1
    def __init__(self, *_a, **_k): pass
    def start(self): pass
    def stop(self): pass
    def read(self, timeout):
        import time
        time.sleep(0.02)
        return np.zeros((64, 1), dtype=np.float32)


def _win(qtbot, tmp_path, session, warning=None):
    from deepfistapp.config import AppConfig
    from deepfistapp.ui.main_window import MainWindow
    cfg = AppConfig(QSettings(str(tmp_path / "w.ini"), QSettings.Format.IniFormat))
    w = MainWindow(cfg, session, startup_warning=warning,
                   devices_fn=_fake_devices,
                   source_factory=lambda device_id: _IdleSource())
    qtbot.addWidget(w)
    return w


def test_window_builds_with_all_spec_elements(qtbot, tmp_path, session):
    w = _win(qtbot, tmp_path, session)
    assert w.windowTitle() == "DeepFistTheApp"
    for attr in ("waterfall", "spectrum", "meters", "transcript", "stats_bar",
                 "transport", "banner", "drawer", "status_chip",
                 "btn_auto", "btn_manual", "source_combo"):
        assert getattr(w, attr) is not None, attr
    assert w.transport.isVisibleTo(w) is False        # hidden outside WAV mode
    assert w.menuBar().actions()                       # File/Audio/Model/Help


def test_startup_warning_shows_banner(qtbot, tmp_path, session):
    w = _win(qtbot, tmp_path, session, warning="custom model rejected — using bundled model")
    assert "bundled" in w.banner.text()


def test_listen_toggle_starts_and_stops_worker(qtbot, tmp_path, session):
    w = _win(qtbot, tmp_path, session)
    w.toggle_listen()                     # replaces w.worker, then starts it
    qtbot.waitUntil(lambda: w.worker.isRunning(), timeout=5000)
    worker = w.worker                     # keep the ref toggle_listen stops
    w.toggle_listen()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)


def test_manual_mode_click_sets_tone(qtbot, tmp_path, session):
    w = _win(qtbot, tmp_path, session)
    w.set_mode_manual(True)
    w.waterfall.tone_clicked.emit(650.0)
    assert w.pipeline_cfg.tone_hz == 650.0
    w.set_mode_manual(False)
    assert w.pipeline_cfg.tone_hz is None


def test_open_bad_wav_shows_error_and_stays_idle(qtbot, tmp_path, session):
    w = _win(qtbot, tmp_path, session)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"NOTAWAV")
    w.open_wav(str(bad))
    assert "not a WAV file" in w.banner.text()
    assert not w.worker.isRunning()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_window.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.ui.main_window'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/ui/main_window.py`:

```python
"""Main window — spec §7.1 layout, §8 state handling."""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QVBoxLayout,
                               QWidget)

from deepfistapp import APP_NAME, theme
from deepfistapp.audio.devices import list_output_devices
from deepfistapp.audio.loopback import LoopbackSource
from deepfistapp.audio.source import SourceError
from deepfistapp.audio.wav_source import WavError, WavSource
from deepfistapp.engine.pipeline import PipelineConfig
from deepfistapp.engine.session import (ModelError, OnnxSession,
                                        bundled_model_path, load_with_fallback)
from deepfistapp.engine.states import AppState
from deepfistapp.engine.worker import DecoderWorker
from deepfistapp.ui.banners import Banner
from deepfistapp.ui.meters import MetersWidget
from deepfistapp.ui.settings_drawer import SettingsDrawer
from deepfistapp.ui.spectrum import SpectrumWidget
from deepfistapp.ui.stats_bar import StatsBar
from deepfistapp.ui.transcript import TranscriptView
from deepfistapp.ui.transport import TransportStrip
from deepfistapp.ui.waterfall import WaterfallWidget

_CHIP = {AppState.IDLE: "■ STOPPED", AppState.LISTENING: "● LISTENING",
         AppState.PLAYING: "▶ PLAYING", AppState.PAUSED: "⏸ PAUSED",
         AppState.DEVICE_LOST: "⚠ DEVICE LOST",
         AppState.RESCANNING: "⟳ RESCANNING"}


class MainWindow(QMainWindow):
    def __init__(self, config, session, startup_warning=None,
                 devices_fn=list_output_devices, source_factory=None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.cfg = config
        self.session = session
        self.devices_fn = devices_fn
        self.source_factory = source_factory or \
            (lambda device_id: LoopbackSource(device_id))
        self.pipeline_cfg = PipelineConfig(
            squelch_thresh=config.squelch_thresh)
        self.wav_source = None
        self._manual = False
        self._build_ui()
        self._build_menus()
        self._make_worker()
        self._rescan_timer = QTimer(self, interval=2000)
        self._rescan_timer.timeout.connect(self._try_rescan)
        if startup_warning:
            self.banner.show_message(startup_warning)

    # ---- construction -------------------------------------------------
    def _build_ui(self):
        self.banner = Banner()
        self.status_chip = QPushButton(_CHIP[AppState.IDLE])
        self.status_chip.clicked.connect(self.toggle_listen)
        self.source_combo = QComboBox()
        self._fill_devices()
        self.btn_open_wav = QPushButton("Open WAV")
        self.btn_open_wav.clicked.connect(self._pick_wav)
        self.btn_auto = QPushButton("AUTO TRACK", checkable=True, checked=True)
        self.btn_manual = QPushButton("MANUAL", checkable=True)
        self.btn_auto.clicked.connect(lambda: self.set_mode_manual(False))
        self.btn_manual.clicked.connect(lambda: self.set_mode_manual(True))
        self.btn_settings = QPushButton("⚙ Settings")
        self.waterfall = WaterfallWidget()
        self.spectrum = SpectrumWidget()
        self.meters = MetersWidget()
        self.transcript = TranscriptView()
        self.stats_bar = StatsBar()
        self.transport = TransportStrip()
        self.drawer = SettingsDrawer(self.cfg, devices_fn=self.devices_fn)
        self.btn_settings.clicked.connect(self.drawer.toggle)
        self.waterfall.tone_clicked.connect(self._on_waterfall_click)
        self.transport.play_pause.connect(self._wav_play_pause)
        self.transport.stop.connect(self._stop_worker)
        self.transport.seek.connect(self._wav_seek)
        self.drawer.changed.connect(self._on_setting_changed)
        self.drawer.rescan_requested.connect(self._fill_devices)
        self.drawer.load_model_requested.connect(self._pick_model)
        self.drawer.revert_model_requested.connect(self.revert_model)

        top = QHBoxLayout()
        top.addWidget(QLabel(APP_NAME))
        top.addWidget(self.source_combo, 1)
        for b in (self.btn_open_wav, self.status_chip, self.btn_auto,
                  self.btn_manual, self.btn_settings):
            top.addWidget(b)

        grid = QGridLayout()                       # waterfall over spectrum,
        grid.addWidget(self.waterfall, 0, 0)       # meters span both (spec §7.1)
        grid.addWidget(self.spectrum, 1, 0)
        grid.addWidget(self.meters, 0, 1, 2, 1)
        grid.setColumnStretch(0, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(self.stats_bar, 1)

        root = QVBoxLayout()
        root.addWidget(self.banner)
        root.addLayout(top)
        root.addLayout(grid)
        root.addWidget(self.transcript, 1)
        root.addWidget(self.transport)
        root.addLayout(bottom)
        central = QWidget()
        body = QHBoxLayout(central)
        body.setContentsMargins(6, 6, 6, 6)
        inner = QWidget(); inner.setLayout(root)
        body.addWidget(inner, 1)
        body.addWidget(self.drawer)
        self.setCentralWidget(central)

    def _build_menus(self):
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction("Open WAV…", self._pick_wav)
        m_file.addAction("Save Transcript…", self._save_transcript)
        m_file.addAction("Quit", self.close)
        m_audio = self.menuBar().addMenu("&Audio")
        m_audio.addAction("Rescan devices", self._fill_devices)
        m_model = self.menuBar().addMenu("&Model")
        m_model.addAction("Load model…", self._pick_model)
        m_model.addAction("Revert to bundled", self.revert_model)
        m_help = self.menuBar().addMenu("&Help")
        m_help.addAction("macOS audio routing guide", self._show_mac_guide)
        m_help.addAction("About", self._show_about)

    def _make_worker(self):
        self.worker = DecoderWorker(
            self.session, self.pipeline_cfg,
            tick_s=self.cfg.tick_s, guard_s=self.cfg.guard_s,
            wpm_bounds=(self.cfg.wpm_min, self.cfg.wpm_max))
        self.worker.text_committed.connect(self.transcript.append_text)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.spectra_row.connect(self._on_row)
        self.worker.state_changed.connect(self._on_state)

    # ---- devices / sources -------------------------------------------
    def _fill_devices(self):
        self.source_combo.clear()
        for d in self.devices_fn():
            label = f"{d.name}{'  (default)' if d.is_default else ''}"
            self.source_combo.addItem(label, d.id)
        self.drawer.refresh_devices()

    def _selected_device_id(self):
        if self.cfg.follow_default:
            return None
        return self.source_combo.currentData()

    # ---- actions ------------------------------------------------------
    def toggle_listen(self):
        if self.worker.isRunning():
            self._stop_worker()
            return
        self._make_worker()
        try:
            src = self.source_factory(self._selected_device_id())
        except SourceError as e:
            self.banner.show_message(str(e))
            return
        self.worker.attach_source(src, AppState.LISTENING)
        self.transport.setVisible(False)
        self.worker.start()

    def open_wav(self, path: str):
        try:
            self.wav_source = WavSource(path)
        except WavError as e:
            self.banner.show_message(str(e))
            return
        if self.worker.isRunning():
            self._stop_worker()
        self._make_worker()
        self.worker.attach_source(self.wav_source, AppState.PLAYING)
        self.transport.setVisible(True)
        self.transport.set_playing(True)
        self._pos_timer = QTimer(self, interval=250)
        self._pos_timer.timeout.connect(lambda: self.transport.set_position(
            self.wav_source.position_s, self.wav_source.duration_s))
        self._pos_timer.start()
        self.worker.start()

    def _pick_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open WAV", "",
                                              "WAV files (*.wav)")
        if path:
            self.open_wav(path)

    def _wav_play_pause(self):
        if self.wav_source is None:
            return
        self._paused = not getattr(self, "_paused", False)
        self.wav_source.paused = self._paused         # source stops feeding
        self.worker.set_paused(self._paused)          # PLAYING <-> PAUSED
        self.transport.set_playing(not self._paused)

    def _wav_seek(self, seconds: float):
        if self.wav_source is not None:
            self.wav_source.seek(seconds)
            self.worker.set_model(self.session)       # engine reset (spec §4)

    def _stop_worker(self):
        self.worker.request_stop()
        self.worker.wait(5000)

    def set_mode_manual(self, manual: bool):
        self._manual = manual
        self.btn_auto.setChecked(not manual)
        self.btn_manual.setChecked(manual)
        if not manual:
            self.pipeline_cfg.tone_hz = None
            self.waterfall.set_marker(None, manual=False)

    def _on_waterfall_click(self, hz: float):
        if self._manual:
            self.pipeline_cfg.tone_hz = float(hz)
            self.waterfall.set_marker(hz, manual=True)

    def load_model(self, path: str):
        try:
            new = OnnxSession(path)
        except ModelError as e:
            self.banner.show_message(f"model rejected: {e}")
            return
        self.session = new
        self.cfg.model_path = str(path)
        self.worker.set_model(new)
        self.transcript.add_marker(f"— model: {new.name} —")
        self.drawer.set_model_info(new.name, str(new.path), is_bundled=False)
        if new.warning:
            self.banner.show_message(new.warning)

    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ONNX model", "",
                                              "ONNX models (*.onnx)")
        if path:
            self.load_model(path)

    def revert_model(self):
        session, warn = load_with_fallback(None, bundled_model_path())
        self.session = session
        self.cfg.model_path = None
        self.worker.set_model(session)
        self.transcript.add_marker(f"— model: {session.name} —")
        self.drawer.set_model_info(session.name, str(session.path),
                                   is_bundled=True)

    # ---- worker feedback ---------------------------------------------
    def _on_state(self, state):
        self.status_chip.setText(_CHIP[state])
        if state is AppState.DEVICE_LOST:
            self.banner.show_message(
                "output device disconnected — waiting for it to return",
                persistent=True)
            if self.cfg.auto_resume:
                self._rescan_timer.start()
        elif state in (AppState.LISTENING, AppState.PLAYING):
            self._rescan_timer.stop()
            self.banner.clear()

    def _try_rescan(self):
        wanted = self._selected_device_id()
        ids = [d.id for d in self.devices_fn()]
        if wanted is None and ids or wanted in ids:
            self._rescan_timer.stop()
            self.toggle_listen()                      # rebuild worker + source

    def _on_stats(self, stats):
        self.stats_bar.update_stats(stats)
        self.meters.set_levels(stats["l_db"], stats["r_db"])
        self.drawer.set_live_score(stats["score"])
        self.drawer.set_diagnostics(stats)
        if stats.get("active") and not self._manual and stats.get("tone_hz"):
            self.waterfall.set_marker(stats["tone_hz"], manual=False)

    def _on_row(self, row):
        self.waterfall.add_row(row)
        self.spectrum.set_row(row)

    def _on_setting_changed(self, key, value):
        if key == "squelch/thresh":
            self.worker.set_squelch(value)
        elif key == "display/wf_floor_db":
            self.waterfall.set_levels(value, self.cfg.wf_gain_db)
        elif key == "display/wf_gain_db":
            self.waterfall.set_levels(self.cfg.wf_floor_db, value)

    # ---- misc ---------------------------------------------------------
    def _save_transcript(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript",
                                              "copy.txt", "Text (*.txt)")
        if path:
            self.transcript.save_to(path)

    def _show_mac_guide(self):
        from PySide6.QtWidgets import QMessageBox
        guide = (Path(__file__).resolve().parent.parent / "resources"
                 / "macos_loopback.md").read_text(encoding="utf-8")
        box = QMessageBox(self)
        box.setWindowTitle("macOS audio routing")
        box.setText(guide)
        box.show()                                   # non-blocking

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox
        from deepfistapp import __version__
        box = QMessageBox(self)
        box.setWindowTitle("About")
        box.setText(f"{APP_NAME} {__version__}\nGPL-3.0-or-later\n"
                    f"model: {self.session.name}\n"
                    "Bundled: PySide6 (LGPL-3), onnxruntime (MIT), "
                    "numpy/scipy (BSD), soundcard (BSD)")
        box.show()

    def closeEvent(self, ev):
        if self.worker.isRunning():
            self._stop_worker()
        super().closeEvent(ev)
```

Create `deepfistapp/main.py`:

```python
"""Entry point: deepfist-app [--wav path]."""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from deepfistapp import APP_NAME, ORG_NAME
from deepfistapp.config import AppConfig
from deepfistapp.engine.session import bundled_model_path, load_with_fallback
from deepfistapp.theme import build_qss
from deepfistapp.ui.main_window import MainWindow


def main() -> int:
    ap = argparse.ArgumentParser(prog="deepfist-app")
    ap.add_argument("--wav", default=None, help="open this WAV in test mode")
    args, qt_args = ap.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyleSheet(build_qss())
    cfg = AppConfig()
    session, warning = load_with_fallback(cfg.model_path, bundled_model_path())
    win = MainWindow(cfg, session, startup_warning=warning)
    win.resize(980, 700)
    win.show()
    if args.wav:
        win.open_wav(args.wav)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_ui_window.py -q`
Expected: `5 passed`

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, 0 failures (full suite regression check)

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/ui/main_window.py deepfistapp/main.py tests/app/test_ui_window.py
git commit -m "feat(app): main window wiring + deepfist-app entry point"
```

---

### Task 18: Integration tests — golden WAV parity, silence, overload

**Files:**
- Create: `deepfistapp/engine/offline.py`
- Test: `tests/app/test_integration.py`

**Interfaces:**
- Consumes: `WavSource` (8), `OnnxSession` (10), `PipelineConfig`/`process_window` (11), `render_text` (4), torch reference modules (test-only).
- Produces: `decode_wav_offline(path: str, session, cfg: PipelineConfig) -> str` — whole-file single-window decode (mono-mix, resample to 3200, full pipeline, full greedy decode, `render_text`). Used by tests and the packaging launch check.

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_integration.py`:

```python
"""Spec §10.3: golden parity, silence, overload. Acceptance criteria 2, 3, 10."""
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepfist.morse.alphabet import TOKENS  # noqa: E402

GOLDEN_TEXT = "CQ CQ DE N9BC N9BC K"


def _synth_wav(tmp_path, text=GOLDEN_TEXT, wpm=20):
    from deepfist.synth.generator import _render_cw
    rng = np.random.default_rng(3)
    sr = 3200
    clip = _render_cw(rng, text, wpm, pitch=650.0, sr=sr,
                      n=int(12 * sr), drift_max=0.0)
    p = tmp_path / "golden.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(clip, -1, 1) * 32767).astype("<i2").tobytes())
    return p, clip.astype(np.float32)


@pytest.fixture(scope="module")
def rand_session(tmp_path_factory):
    from deepfist.export.to_onnx import export_onnx, write_metadata
    from deepfist.model.net import CwCtcNet
    from deepfistapp.engine.session import OnnxSession
    d = tmp_path_factory.mktemp("m")
    net = CwCtcNet(width=1.0); net.eval()
    export_onnx(net, str(d / "m.onnx")); write_metadata(str(d / "m.onnx"))
    torch.save(net.state_dict(), d / "model.pt")
    return OnnxSession(d / "m.onnx"), d


def _torch_reference_text(ckpt_dir, audio):
    """Same algorithm, torch features + torch net (the parity reference)."""
    import json
    from deepfist.dsp.despike import despike
    from deepfist.features.conditioner import condition
    from deepfist.features.spectrogram import audio_to_spectrogram
    from deepfist.model.net import CwCtcNet
    from deepfistapp.engine.commit import greedy_frames, render_text
    cfg_p = ckpt_dir / "config.json"
    kw = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    net = CwCtcNet(width=kw.get("width", 1.0),
                   time_downsample=kw.get("time_downsample", 2))
    net.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    net.eval()
    cond = condition(despike(audio, 3200), 3200)
    spec = audio_to_spectrogram(cond, 3200).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        lp = net(spec).numpy()
    ids, _f, _n = greedy_frames(lp)
    return render_text(ids)


def test_golden_wav_parity_random_model(tmp_path, rand_session):
    """App engine text == torch reference text, whatever the (random) model says."""
    from deepfistapp.engine.offline import decode_wav_offline
    from deepfistapp.engine.pipeline import PipelineConfig
    session, d = rand_session
    p, clip = _synth_wav(tmp_path)
    app_text = decode_wav_offline(str(p), session, PipelineConfig())
    ref_text = _torch_reference_text(d, clip)
    assert app_text == ref_text


CHAMPION = "runs/exp27_bt/model.pt"


@pytest.mark.skipif(not __import__("pathlib").Path(CHAMPION).exists(),
                    reason="champion checkpoint not present")
def test_golden_wav_real_text_with_champion(tmp_path):
    from deepfist.export.to_onnx import export_from_checkpoint
    from deepfistapp.engine.offline import decode_wav_offline
    from deepfistapp.engine.pipeline import PipelineConfig
    from deepfistapp.engine.session import OnnxSession
    out = tmp_path / "champ.onnx"
    export_from_checkpoint(CHAMPION, str(out))
    p, _clip = _synth_wav(tmp_path)
    text = decode_wav_offline(str(p), OnnxSession(out), PipelineConfig())
    assert text.replace(" ", "") == GOLDEN_TEXT.replace(" ", "")


def test_60s_silence_emits_zero_chars(tmp_path, rand_session):
    """Acceptance criterion 3 — digital silence and low noise."""
    from deepfistapp.engine.offline import decode_wav_offline
    from deepfistapp.engine.pipeline import PipelineConfig
    session, _d = rand_session
    rng = np.random.default_rng(1)
    for sig in (np.zeros(60 * 3200, dtype=np.float32),
                (0.01 * rng.standard_normal(60 * 3200)).astype(np.float32)):
        p = tmp_path / "quiet.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(3200)
            w.writeframes((sig * 32767).astype("<i2").tobytes())
        assert decode_wav_offline(str(p), session, PipelineConfig()) == ""


def test_overload_drops_do_not_crash_and_are_counted(qtbot, rand_session):
    """Acceptance criterion 10 — flood the queue far past its bound."""
    from deepfistapp.engine.pipeline import PipelineConfig
    from deepfistapp.engine.worker import DecoderWorker
    session, _d = rand_session
    w = DecoderWorker(session, PipelineConfig())
    block = np.zeros((320, 1), dtype=np.float32)
    for _ in range(500):
        w.submit_block(block)
    assert w.overruns == 500 - w.QUEUE_MAX
    assert len(w._queue) == w.QUEUE_MAX
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_integration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'deepfistapp.engine.offline'`

- [ ] **Step 3: Write minimal implementation**

Create `deepfistapp/engine/offline.py`:

```python
"""Offline whole-file decode through the exact live pipeline (test/verify path)."""
import numpy as np
from scipy.signal import resample_poly

from deepfistapp.audio.wav_source import WavSource
from deepfistapp.engine.commit import render_text
from deepfistapp.engine.numpy_features import SAMPLE_RATE
from deepfistapp.engine.pipeline import process_window


def decode_wav_offline(path: str, session, cfg) -> str:
    src = WavSource(path, realtime=False)
    src.start()
    blocks = []
    while True:
        b = src.read(timeout=1.0)
        if b is None:
            break
        blocks.append(b.mean(axis=1) if b.ndim > 1 else b)
    src.stop()
    if not blocks:
        return ""
    audio = np.concatenate(blocks).astype(np.float32)
    if src.samplerate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, src.samplerate).astype(np.float32)
    res = process_window(audio, session, cfg)
    if not res.active:
        return ""
    return render_text(res.ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/app/test_integration.py -q`
Expected: `4 passed` (5 with the champion checkpoint present; the champion test
must pass on this machine, where `runs/exp27_bt/model.pt` exists)

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, 0 failures — full-suite regression gate

- [ ] **Step 5: Commit**

```bash
git add deepfistapp/engine/offline.py tests/app/test_integration.py
git commit -m "test(app): golden-WAV parity, silence, and overload integration gates"
```

---

### Task 19: Packaging, cross-platform checks, end-user launch verification

**Files:**
- Create: `packaging/deepfist_app.spec`, `packaging/build_release.py`
- Modify: `README.md` (add a "Desktop app" section after "Quick start")
- Test: manual verification checklist (below) — this task has no new pytest file

**Interfaces:**
- Consumes: `deepfist.export.to_onnx.export_from_checkpoint`, champion checkpoint `runs/exp27_bt/model.pt`, `decode_wav_offline` (Task 18).

- [ ] **Step 1: Write the PyInstaller spec**

Create `packaging/deepfist_app.spec`:

```python
# PyInstaller one-dir GUI build for DeepFistTheApp (run from repo root).
# torch is explicitly excluded — the app must run without it.
from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "deepfistapp" / "main.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "deepfistapp" / "resources"), "deepfistapp/resources")],
    hiddenimports=["soundcard"],
    excludes=["torch", "torchaudio", "torchvision", "matplotlib", "PIL"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, exclude_binaries=True,
          name="DeepFistTheApp", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DeepFistTheApp")
```

- [ ] **Step 2: Write the release builder**

Create `packaging/build_release.py`:

```python
"""Build a distributable DeepFistTheApp: export the champion model into
resources (build-time artifact — resources/*.onnx is git-ignored), then run
PyInstaller. Usage:  .venv/Scripts/python.exe packaging/build_release.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs" / "exp27_bt" / "model.pt"
OUT = ROOT / "deepfistapp" / "resources" / "deepfist.onnx"


def main() -> int:
    if not CKPT.exists():
        print(f"champion checkpoint missing: {CKPT}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    from deepfist.export.to_onnx import export_from_checkpoint
    export_from_checkpoint(str(CKPT), str(OUT))
    print(f"exported {OUT} + sidecar")
    return subprocess.call([sys.executable, "-m", "PyInstaller", "--noconfirm",
                            str(ROOT / "packaging" / "deepfist_app.spec")])


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: README section + install PyInstaller**

Append to `README.md` directly after the "Quick start" section:

```markdown
## Desktop app (DeepFistTheApp)

A PySide6 GUI that decodes whatever your computer is playing (WASAPI loopback /
PipeWire monitor / BlackHole on macOS) or a WAV file — no radio required.

    pip install -e ".[app]"
    deepfist-app                # or: python -m deepfistapp.main
    deepfist-app --wav test.wav # WAV test mode

Build a standalone bundle (needs the champion checkpoint under `runs/`):

    pip install pyinstaller
    python packaging/build_release.py   # dist/DeepFistTheApp/

The design of record is `docs/superpowers/specs/2026-07-19-deepfist-the-app-design.md`.
```

Run: `.venv/Scripts/python.exe -m pip install pyinstaller`
Expected: exit 0.

- [ ] **Step 4: Build and verify the Windows bundle**

Run: `.venv/Scripts/python.exe packaging/build_release.py`
Expected: `exported ...deepfist.onnx + sidecar`, then PyInstaller completes with
`Building COLLECT COLLECT-00.toc completed successfully` and `dist/DeepFistTheApp/DeepFistTheApp.exe` exists.

Run: `ls dist/DeepFistTheApp/ | head` and verify `DeepFistTheApp.exe` plus
`_internal/deepfistapp/resources/deepfist.onnx` are present.

Run (bundle must not contain torch): `ls dist/DeepFistTheApp/_internal | grep -i torch || echo "no torch - OK"`
Expected: `no torch - OK`

- [ ] **Step 5: End-user launch verification (final acceptance)**

Dev-mode launch:

- [ ] Run `.venv/Scripts/python.exe -m deepfistapp.main` → window titled
  **DeepFistTheApp** appears with dark Lyra theme, `■ STOPPED` chip, device
  selector populated, all panels visible, settings drawer toggles.
- [ ] Generate a golden WAV
  (`.venv/Scripts/python.exe -c "from tests.app.test_integration import _synth_wav; from pathlib import Path; _synth_wav(Path('.'))"`
  creates `golden.wav`), then `.venv/Scripts/python.exe -m deepfistapp.main --wav golden.wav`
  → transport strip appears, status chip `▶ PLAYING`, decoded text streams into
  the transcript and reads **CQ CQ DE N9BC N9BC K** (spaces may vary), chip
  returns to `■ STOPPED` at EOF. Copy puts text on the clipboard; Save writes it;
  Clear needs the two-click confirm.
- [ ] With no CW playing anywhere (live mode on a silent device): press the
  status chip → `● LISTENING`; the transcript stays **empty** (no-hallucination
  rule observed end-to-end).

Packaged launch (same checks against the bundle):

- [ ] `dist/DeepFistTheApp/DeepFistTheApp.exe` cold start ≤ 5 s to interactive
  (acceptance criterion 11), then the `--wav golden.wav` decode check above.

Cross-platform (documented commands — run on those machines; not executable on
this Windows box):

- [ ] Linux (Ubuntu 24.04/PipeWire): `pip install -e ".[app]" && deepfist-app`;
  select a "Monitor of …" source; play any CW video/file through the sink;
  text appears. `python packaging/build_release.py` for the tar.gz payload.
- [ ] macOS 14 (BlackHole installed per the in-app guide): select BlackHole as
  the source; same golden-WAV + live checks; `.app` assembly per spec §13 is a
  release-procedure step, not part of this plan's automated scope.

- [ ] **Step 6: Full-suite final gate + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, 0 failures (baseline 109 + every app test added above).

```bash
git add packaging README.md
git commit -m "feat(app): PyInstaller packaging, release builder, launch verification"
```

---

## Self-Review (performed on this plan)

1. **Spec coverage** — every spec section maps to a task: §2 hard rules →
   Tasks 4/11/13/18 (order test, gate short-circuit, silence gates); §3.1
   promotion → Task 1; §3.2 layout/boundaries → Tasks 2–17 (scaffold test
   enforces no-torch); §3.3 numpy parity → Task 3; §4 queue/threading →
   Tasks 13/18; §5.1 loopback/devices → Task 9; §5.2 WAV + rejections →
   Task 8; §6 Auto/Manual → Tasks 11/17; §7.1 layout + widgets → Tasks
   14–17; §7.2 exact tokens → Task 6; §7.3 drawer → Task 16; §8 states +
   disconnect/rescan → Tasks 5/13/17; §9 model management → Tasks 10/17;
   §10 tests → Tasks 1–18 test files; §11 acceptance criteria → automated
   in Tasks 3/13/18 + manual in Task 19; §12 macOS guidance → Tasks 2/17;
   §13 packaging → Tasks 2/19. Latency/FPS criteria (4, 6) are verified via
   the diagnostics readout during Task 19's launch checks.
2. **Placeholder scan** — no TBD/TODO/"similar to"/"add error handling"
   phrases; every code step contains the actual code; every Run step has an
   expected outcome.
3. **Type consistency** — signatures cross-checked: `has_signal` tuple
   return (T1→T11), `WindowResult` fields (T11→T13), stats dict keys
   (T13→T15/T16/T17), `AudioSource.read -> ndarray | None` (T8→T9/T13),
   `OnnxSession.run [1,1,65,T]→[T',1,48]` (T10→T11/T18), signal names
   (`tone_clicked`, `text_committed`, `state_changed`) uniform across
   T13–T17.
4. **Ordering** — strictly bottom-up: promotion → features → engine pieces →
   worker → widgets → window → integration → packaging; no task consumes an
   interface defined later.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-deepfist-the-app.md`.
The user chose **continued inline work in this same Claude Desktop session**,
so execution will use **superpowers:executing-plans** (batch execution with
checkpoints); superpowers:subagent-driven-development is the alternative if a
fresh-context-per-task run is ever preferred.
