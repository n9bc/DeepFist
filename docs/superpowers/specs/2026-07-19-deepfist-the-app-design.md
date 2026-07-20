# DeepFistTheApp — Desktop Application Design

- **Date:** 2026-07-19
- **Status:** Approved (documentation only; no implementation authorized by this document)
- **License:** GPL-3.0-or-later (inherits the DeepFist repository license)
- **Platforms:** Windows, Linux, macOS (x64/arm64 where dependencies provide wheels)

## 1. Overview

DeepFistTheApp is a cross-platform desktop application that turns the DeepFist
neural CW decoder into a standalone "CW reader": point it at whatever your
computer is playing (a WebSDR tab, an SDR application, a rig's audio), click a
signal, and read clean decoded text. It is a **PySide6/Qt** application using
**ONNX Runtime** for inference — no PyTorch dependency at runtime.

The application reuses the proven DeepFist pipeline unchanged and adds only
audio routing, threading, and UI around it.

### Goals

- Live decode of the computer's **system output** (loopback) as the primary source.
- A **WAV test mode** that exercises the *exact same* processing path.
- A UI that visually matches **Lyra-SDR** (shared theme tokens, modern SDR
  layout, restrained glass styling).
- Ship with a **bundled default ONNX model**; power users may load another.
- Preserve every existing DeepFist behavior, CLI, and test.

### Non-goals (v1)

- No transmit, rig control, or TCI client (the `scripts/tci_decode.py` CLI
  remains the TCI path).
- No training or dataset UI.
- No multi-signal simultaneous decode (one selected signal at a time).
- No language-model post-processing (see HANDOFF §18.24 — violates the
  no-hallucination rule as currently designed).
- No audible monitoring of WAV playback (decode-only; playing audio out while
  capturing system output invites feedback loops).

## 2. Inherited hard rules (non-negotiable)

These are product identity, carried over from DeepFist verbatim:

1. **No signal energy → no characters.** Silence must never produce invented
   text. The squelch gate runs on **raw** audio; a closed gate skips the entire
   downstream pipeline for that window.
2. **Fixed processing order:**
   `raw-audio squelch → despike → condition → spectrogram → ONNX inference`.
   Conditioning is mandatory before the model and must never run before the
   squelch (its AGC/bandpass fabricates a tone from noise).
3. **All input is converted to mono 3200 Hz** (`deepfist.features.spectrogram.SAMPLE_RATE`)
   before any processing: multi-channel is mean-downmixed, then resampled with
   `scipy.signal.resample_poly` (or `decimate` for integer factors).
4. Punctuation suppression (`,` and `.` stripped) and BT (`=`) rendered as a
   line break are the default display transforms, matching `tci_decode.py`.
5. Blank penalty is 0 for the conditioned pipeline.

## 3. Architecture & module boundaries

### 3.1 Library promotion (prerequisite, semantics-preserving)

`tools/squelch.py` and `tools/despike.py` are currently outside the installable
package and reached via a `sys.path` hack. They move into a new subpackage:

```
deepfist/dsp/
  __init__.py
  squelch.py      # moved verbatim from tools/squelch.py
  despike.py      # moved verbatim from tools/despike.py
  tempo.py        # estimate_wpm moved from tools/tempo.py (needed for the WPM statistic)
```

Rules for the move:

- **No semantic changes.** Function names, signatures, defaults
  (`DEFAULT_THRESH = 12`), and numeric behavior are unchanged; the move is
  `git mv` plus import-path fixes only.
- `tools/squelch.py`, `tools/despike.py`, and `tools/tempo.py` become one-line
  re-export shims (`from deepfist.dsp.squelch import *  # noqa`, preserving
  module-level names such as `has_signal` and `DEFAULT_THRESH`) so every
  existing script, tool, and test import keeps working.
- `tempo.py` moves only because the app's WPM statistic needs `estimate_wpm`
  from the installed package; the rejected tempo-*normalization* path stays
  rejected and is not exposed in the app.
- Existing tests must pass unmodified after the move; a new unit test asserts
  shim imports resolve to the same objects as package imports.

### 3.2 Application package

The app lives in the same repository as a sibling top-level package:

```
deepfistapp/
  __init__.py
  main.py                 # entry point: QApplication, MainWindow, --wav CLI arg
  theme.py                # Lyra token constants + generated QSS (see §7.2)
  config.py               # QSettings-backed persisted settings
  audio/
    source.py             # AudioSource protocol: start/stop/read, device id, sr, channels
    devices.py            # enumeration + monitoring of output endpoints per OS
    loopback.py           # system-output capture via the `soundcard` library
    wav_source.py         # file source implementing the same AudioSource protocol
  engine/
    pipeline.py           # fixed-order window processor (squelch→despike→condition→spec→ONNX)
    session.py            # ONNX Runtime session wrapper + model validation/metadata
    worker.py             # decoder thread: ring buffer, bounded queue, commit logic
    spectra.py            # display FFT tap for spectrum/waterfall rows
    numpy_features.py     # numpy port of audio_to_spectrogram (see §3.3)
  ui/
    main_window.py
    waterfall.py          # clickable waterfall widget
    spectrum.py           # spectrum trace widget
    meters.py             # L/R level meters
    transcript.py         # decoded-text area + Copy/Save/Clear
    stats_bar.py          # WPM / signal / tone / model readouts
    settings_drawer.py    # sliding settings panel
    transport.py          # WAV transport controls (hidden outside WAV mode)
    banners.py            # non-blocking notification strip (device lost, fallback, overrun)
  resources/
    deepfist.onnx         # bundled default model — build-time artifact,
                          #   not committed (repo ignores *.onnx); see §13
    deepfist.onnx.json    # its metadata sidecar (written by deepfist.export)
    macos_loopback.md     # in-app macOS setup guide content
```

Boundaries:

- `deepfistapp.engine` may import from `deepfist.*`; `deepfistapp.ui` may
  import from `deepfistapp.engine` but never from `deepfist.*` directly.
- `deepfist.*` never imports from `deepfistapp.*`.
- Nothing in `deepfistapp` imports `torch`. App runtime dependencies:
  `PySide6`, `onnxruntime`, `numpy`, `scipy`, `soundcard`.

### 3.3 Spectrogram without torch

`deepfist.features.spectrogram.audio_to_spectrogram` uses torch STFT. The app
uses a numpy implementation (`engine/numpy_features.py`) reproducing the exact
documented preprocessing (Hann window, `N_FFT`/`HOP`, band crop
`BAND_LO_HZ`–`BAND_HI_HZ`, `log1p` compression, global standardization — the
same parameters the ONNX metadata sidecar records). A regression test asserts
max absolute difference ≤ 1e-4 versus the torch reference over a fixed test
corpus, and an integration test asserts **identical decoded text** on golden
WAVs (§10.3). If parity cannot be met, the fallback is declaring torch an app
dependency — a packaging cost, never a silent accuracy change.

## 4. Data flow & threading

```
capture callback (soundcard thread)          UI thread (Qt main loop)
  raw float32 blocks, native sr/ch                 ▲
        │                                          │ Qt signals (queued)
        ▼                                          │
  bounded block queue  (maxsize = 32 blocks)       │
        │                                          │
        ▼                                          │
  DecoderWorker (QThread) ── every tick (0.4 s): ──┤
    downmix → resample → append to 6 s ring        │  • text_committed(str)
    squelch(raw ring)  ──closed──► advance commit  │  • stats_updated(dict)
        │ open                     boundary only   │  • spectra_row(ndarray)
        ▼                                          │  • state_changed(enum)
    despike → condition → numpy spectrogram        │
        ▼                                          │
    ONNX session.run → greedy CTC + frame times    │
        ▼                                          │
    guard-delay commit logic (ported verbatim      │
    from scripts/tci_decode.py) ───────────────────┘
```

- The **capture thread** only copies blocks into the queue — no processing, no
  allocation beyond the block, never blocks on the UI.
- The **DecoderWorker** owns the ring buffer, all DSP, inference, and the
  commit boundary (`committed_t`), and emits only via queued Qt signals.
- The **UI thread** does rendering only: transcript append, waterfall row
  blit, meter/stat updates. No DSP or file I/O on the UI thread; Save uses a
  brief modal file dialog and a synchronous write of the (small) transcript.
- The spectrum/waterfall tap (`spectra.py`) computes one 512-point FFT row per
  display frame (target 20 Hz) from the decimated mono ring — decoupled from
  the decode tick so display stays smooth even while inference runs.

### Bounded-queue overload & reset behavior

- Queue `maxsize` 32 blocks (~3 s of audio at typical 100 ms blocks). On a full
  queue the capture callback **drops the oldest block** (ring semantics),
  increments an `overruns` counter (visible in Diagnostics), and never blocks.
- If the worker detects that its input ring lost continuity (dropped blocks),
  it advances `committed_t` past the discontinuity so stale window content is
  never re-emitted as fresh text.
- **Reset** (source change, device change, WAV seek, model swap, Stop): flush
  queue, zero ring, reset `committed_t` and stats. The transcript is *not*
  cleared by resets — only by the user's Clear.
- Sustained overload (queue >75 % for >10 s) raises a persistent banner
  ("Decoder can't keep up — close other apps or raise the decode interval in
  Settings → Diagnostics").

## 5. Audio sources

### 5.1 System-output loopback (primary live source)

The user picks an **output device** (render endpoint) from a selector; the app
captures what that device is playing:

- **Windows:** WASAPI loopback. The `soundcard` library exposes every render
  endpoint as a loopback-capable microphone
  (`soundcard.all_microphones(include_loopback=True)`); the selector maps
  render endpoint → its loopback capture. Default selection: "Follow system
  default output".
- **Linux:** PipeWire/PulseAudio **monitor sources** ("Monitor of …"), which
  both servers expose as normal capture devices. The selector lists each sink's
  monitor; default: monitor of the default sink. PipeWire's Pulse
  compatibility layer makes one code path serve both.
- **macOS:** Core Audio has **no native loopback**. The selector lists any
  installed virtual-audio endpoint (e.g. BlackHole 2ch, Loopback.app, or a
  Multi-Output Device that includes one). If none is present, the selector
  shows a "Set up audio routing…" entry that opens the in-app guide (§12).

Device rows show a live level glyph so the user can see at a glance which
device is actually carrying audio.

### 5.2 WAV test mode

**Open WAV** feeds a file through the *identical* `AudioSource` protocol and
the identical pipeline — same downmix, same resample, same squelch and decode
path; only the byte source differs. Playback is paced at real time to mimic
live behavior and is **decode-only** — the audio is not played out to any
device (per the §1 non-goal). While a WAV is loaded, a compact **transport strip** appears:
play/pause toggle, stop, a seek bar, and `elapsed / total` time. Seeking
performs a reset (§4) and resumes cleanly at the new position. Closing the WAV
(or starting live capture) hides the transport and returns to the live source.

**Accepted:** RIFF/WAVE containing PCM (8/16/24/32-bit int) or IEEE float32,
any sample rate, 1–8 channels.
**Rejected with a specific, non-blocking error dialog** (app stays in its
current state): non-WAV files, WAV with compressed codecs (ADPCM, µ-law, MP3-
in-WAV), zero-length or truncated data chunks, unreadable headers, or files
over 2 GB ("split the recording"). The dialog names the actual reason, e.g.
"`x.wav` uses ADPCM compression — export as 16-bit PCM."

## 6. Decode modes: Auto Track and Manual

- **Auto Track (default):** the conditioner's `detect_tone` picks the
  strongest in-band tone each window (today's pipeline behavior;
  `condition(audio, sr, tone_hz=None)`). The waterfall shows a follow marker
  at the detected pitch.
- **Manual:** the user **clicks a signal on the waterfall**; the clicked
  frequency is passed as the `tone_hz` override to `condition()`, locking the
  front-end to that signal until re-clicked or switched back to Auto. A dial
  marker (Lyra's `#ffaa50` line style) shows the locked frequency; click-drag
  fine-tunes it. Switching modes is a pipeline parameter change only — no
  reset, no transcript interruption.

The mode toggle is a two-state button pair in the toolbar (`AUTO TRACK` /
`MANUAL`), styled like Lyra mode buttons.

## 7. User interface

### 7.1 Layout (single main window, modern SDR arrangement)

```
┌──────────────────────────────────────────────────────────────────┐
│ Menu bar: File · Audio · Model · Help                            │
├──────────────────────────────────────────────────────────────────┤
│  DeepFistTheApp   [source selector ▾] [Open WAV] [● LISTENING]   │
│                   [AUTO TRACK|MANUAL]              [⚙ Settings]  │
├──────────────────────────────────────────────┬───────────────────┤
│  Waterfall (clickable; tone marker;          │  L ▓▓▓▓▓▓░░  −18  │
│  scrolls down)                               │  R ▓▓▓▓▓░░░  −21  │
├──────────────────────────────────────────────┤                   │
│  Spectrum (300–1200 Hz trace, glass panel)   │  (meter strip     │
│                                              │   spans both)     │
├──────────────────────────────────────────────┴───────────────────┤
│  Decoded text (monospace, auto-scroll)                           │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│ WPM 22 · signal 34 (open) · tone 612 Hz · model exp27_bt (bundled)│
│                              [Copy] [Save…] [Clear]              │
└──────────────────────────────────────────────────────────────────┘
   (WAV transport strip appears above the status row in WAV mode)
```

- **Menu bar:** File (Open WAV…, Save Transcript…, Quit) · Audio (device
  submenu mirroring the selector, Rescan devices) · Model (Bundled model,
  Load model…, model info) · Help (macOS audio routing guide, About — GPL
  notice + third-party licenses).
- **Status chip:** `● LISTENING` (accent green pulse) / `■ STOPPED` /
  `▶ PLAYING` / `⏸ PAUSED` / `⚠ DEVICE LOST`; clicking it toggles
  start/stop of the current source.
- **Meters:** L/R peak+RMS of the **raw capture** (pre-downmix), −60…0 dBFS,
  Lyra meter styling; a mono source mirrors both bars.
- **Statistics row:** WPM (from `deepfist.dsp.tempo.estimate_wpm`, clamped to
  the configured speed bounds and smoothed over the last 10 s of open-gate
  audio; em-dash when the gate is closed), signal (live keying-ratio squelch
  score + open/closed), tone (current auto-detected or manual-locked Hz),
  model (name + bundled/custom badge).
- **Copy** puts the transcript on the clipboard; **Save…** writes UTF-8 text;
  **Clear** empties the transcript after an inline confirm.

### 7.2 Theme — exact Lyra tokens (not approximations)

Source of truth: `Lyra-SDR-cpp/src/theme.h` (its resolved-literal token table)
plus the glass surfaces used by Lyra's panadapter QML. `deepfistapp/theme.py`
declares these constants and generates the QSS; values are copied exactly:

| Token | Value | Use |
|---|---|---|
| `BG_APP` | `rgb(10,13,18)` | window background |
| `BG_PANEL` | `rgb(17,22,32)` | panels, menus, tooltips |
| `BG_RECESS` | `rgb(16,20,28)` | recessed wells, tabs, status bar |
| `BG_CTRL` | `rgb(22,28,40)` | inputs, combo boxes |
| `ACCENT` | `rgb(0,229,255)` | electric-cyan primary |
| `ACCENT_DIM` | `rgb(0,168,200)` | pressed/checked accents |
| `ACCENT2` | `rgb(57,255,20)` | neon-green secondary (LISTENING pulse, slider hover) |
| `TEXT_PRIMARY` | `rgb(205,217,229)` | body text, decoded text |
| `TEXT_MUTED` | `rgb(138,154,172)` | labels, statistics |
| `TEXT_FAINT` | `rgb(90,112,128)` | disabled |
| `BORDER` | `rgb(30,42,58)` | all 1 px borders |
| `ENGAGED_ORANGE` | `rgb(255,154,60)` | destructive/restore accents (Clear confirm) |
| `SELECTION` | `rgba(0,229,255,80)` | text selection |
| glass panel | `#cc14202a` fill, `#2a4a5a` border | spectrum/waterfall overlay chips |
| glass surface | `#101820` fill, `#2a4a5a` border | solar-style widget strips |
| text-on-glass | `#8fa6ba`, bright `#8fd0ff` | labels over glass |
| outline shadow | `#80000000` / `#cc000000` | text outlines over spectra |
| dial marker | `#ffaa50` | manual tone lock line |
| RX marker | `#a6ff00` | auto-track follow marker |

Widget styling (buttons, sliders, tabs, group boxes, menus, spin boxes,
scrollbars, tooltips) ports Lyra's QSS rules with these tokens, adapted only
where PySide6 widget names differ. **Restrained glass:** glass fills are used
only for overlay chips on the spectrum/waterfall and the settings drawer
backdrop — never for text-bearing primary surfaces (transcript, dialogs),
which stay on solid `BG_PANEL`/`BG_APP`. If Lyra's theme evolves, this table
is re-derived from `theme.h`, not hand-tweaked.

### 7.3 Settings drawer

A right-edge sliding drawer (glass backdrop, solid content cards), toggled by
the ⚙ button; live-applies every control:

- **Routing:** output-device selector (same model as the toolbar), Follow
  system default toggle, Rescan button, auto-resume-on-reconnect toggle.
- **Squelch:** threshold slider 0–50 (default `DEFAULT_THRESH` = 12) with a
  live keying-score meter beside it so the user can calibrate against dead air.
- **Display levels:** waterfall floor/gain sliders, spectrum averaging factor,
  waterfall speed.
- **Speed bounds:** min/max WPM for the WPM statistic (defaults 10–40,
  hard range 5–60).
- **Model:** current model card (name, path, hash, width from metadata),
  Load model…, Revert to bundled.
- **Diagnostics:** queue depth, overrun counter, last decode-tick duration,
  real-time factor, decode interval (`tick`) and commit delay (`guard`)
  advanced controls, session-log-to-file toggle.

Settings persist via `QSettings` under org `DeepFist`, app `DeepFistTheApp`.

## 8. State machine

States: `IDLE`, `LISTENING` (live), `PLAYING` (WAV), `PAUSED` (WAV),
`DEVICE_LOST`, `RESCANNING`.

```
IDLE ── start (live) ──► LISTENING ── stop ──► IDLE
IDLE ── open WAV + play ─► PLAYING ⇄ PAUSED ── stop/EOF ──► IDLE (WAV stays loaded)
LISTENING ── capture error/timeout >1 s ──► DEVICE_LOST
DEVICE_LOST ── auto (every 2 s) ──► RESCANNING ── device back ──► LISTENING
RESCANNING ── device still absent ──► DEVICE_LOST (banner persists)
DEVICE_LOST ── user picks another device ──► LISTENING
any state ── model swap / device change / seek ──► same state, engine reset (§4)
```

### Device disconnect & rescan behavior

- Loss detection: capture read raises, or delivers no blocks for >1 s while
  the stream claims to be running.
- On `DEVICE_LOST`: stop the stream, keep the transcript and stats frozen,
  show a banner ("`<device>` disconnected — waiting for it to return"), and
  poll device enumeration every 2 s.
- If the **same endpoint id** reappears and auto-resume is enabled (default
  on): resume `LISTENING` automatically and clear the banner. If auto-resume
  is off, the banner gains a Resume button.
- "Follow system default" mode instead re-binds to the *new* default output
  on any default-device change (Windows/macOS notification, PulseAudio
  default-sink change), with a 500 ms debounce.
- Manual **Rescan** (toolbar menu, drawer, banner) re-enumerates immediately.

## 9. Model management

- **Bundled default:** `deepfistapp/resources/deepfist.onnx` + its metadata
  sidecar, exported by the existing `deepfist.export.to_onnx` from the current
  champion checkpoint. It ships inside the package/installer, is read-only,
  and is always available as a fallback.
- **Load Model (advanced):** Model menu / Settings drawer. Validation before
  adoption, in order: file exists; ONNX Runtime session creates successfully
  (CPU EP); input tensor named `spectrogram` with shape `[batch,1,65,time]`;
  output `log_probs` with class dimension exactly `len(TOKENS)` = 48; if a
  `.json` sidecar exists its preprocessing block must match the app's
  constants (mismatch → reject with the differing field named); missing
  sidecar → accept with a warning banner ("no metadata — assuming standard
  preprocessing").
- **Failure/fallback:** a model that fails validation is rejected with a
  dialog naming the failed check; the previously active model stays live. If
  the *persisted* custom model is missing or invalid at startup, the app
  silently falls back to the bundled model and shows a one-time banner.
  Decoding never stops on account of a model problem.
- A model swap mid-session performs an engine reset (§4) and stamps the
  transcript with a marker line (`— model: <name> —`).

## 10. Testing

New tests live under `tests/app/` and are `pytest.importorskip`-guarded on
`PySide6`/`onnxruntime`/`soundcard`, so environments without app extras still
run the existing suite untouched.

### 10.1 Unit

- **Promotion equivalence:** `deepfist.dsp.squelch/despike/tempo` produce
  byte-identical outputs to recorded golden values from the pre-move modules;
  `tools/*` shims re-export the same objects.
- **numpy spectrogram parity:** ≤ 1e-4 max-abs-diff vs torch reference
  across the test corpus (clean synth, noisy synth, real capture excerpt).
- **Pipeline order:** processing stages are invoked in the fixed order; a
  closed squelch gate short-circuits despike/condition/inference entirely.
- **Queue:** bounded, drop-oldest on overflow, overrun counter increments,
  discontinuity advances the commit boundary.
- **Commit logic:** the ported guard-delay logic reproduces `tci_decode.py`
  emissions on a recorded log-probs fixture (no dropped or duplicated chars).
- **State machine:** every legal transition in §8; illegal transitions raise.
- **WAV validation:** each rejection class in §5.2 yields its specific error;
  accepted formats decode.
- **Model validation:** each rejection class in §9; fallback selects the
  bundled model.

### 10.2 UI (pytest-qt, offscreen)

- Main window builds with all §7.1 elements present and themed (spot-check
  token values on key widgets).
- Start/stop chip drives worker state; transport appears only in WAV mode;
  seek triggers reset; Copy/Save/Clear act on the transcript (Clear requires
  confirm).
- Waterfall click in Manual mode sets the expected `tone_hz` for the next
  window; mode toggle switches without a reset.
- Settings drawer round-trips every control through `QSettings`.

### 10.3 Integration

- **Golden WAV end-to-end:** feed reference WAVs (clean synth at 15/25 WPM, a
  real ARRL excerpt) through the full app engine (headless, no UI); decoded
  text must equal the recorded output of the torch reference pipeline exactly.
- **Silence:** 60 s of dead-air capture (from `runs/noise_ref`) and 60 s of
  digital silence each produce **zero** emitted characters.
- **Device-loss drill (scripted, Linux):** create a PulseAudio/PipeWire null
  sink, stream a synth WAV into it, decode from its monitor, kill and
  recreate the sink → app reaches `DEVICE_LOST` then auto-resumes without
  crash or duplicate text. Runnable locally; CI wiring is out of scope for
  this spec. (Windows/macOS: scripted manual checklist in the release
  procedure.)
- **Overload:** artificially slow the worker (sleep injection) → overruns
  counted, banner raised, no crash, recovery within 2 ticks of load removal.

### 10.4 Regression (existing DeepFist)

- The entire pre-existing test suite passes with **zero modifications** to
  the tests themselves (109 tests at the time of writing).
- `scripts/tci_decode.py --seconds 5` still runs against a live/fake TCI
  endpoint using the `tools/` shims (smoke-tested in CI with a stub server).
- `deepfist.export.to_onnx` output is byte-compatible with what Lyra consumes
  (existing `tests/export` suite unchanged).

## 11. Measurable acceptance criteria

1. All pre-existing DeepFist tests pass unmodified (109/109 at spec time).
2. Golden-WAV decode through the app engine is **string-identical** to the
   torch reference pipeline output for all reference WAVs (§10.3).
3. 60 s of below-threshold input (dead air and digital silence) emits **0**
   characters.
4. End-to-end committed-text latency in live mode ≤ `guard + tick + 0.5 s`
   (≤ 2.2 s at defaults), measured by the diagnostics timestamps.
5. Each decode tick completes in < 0.2 s — half the 0.4 s tick interval —
   with the bundled model on the **reference machine**: a 4-core x86-64 CPU
   (2020-era), no GPU. Criteria 6 and 11 use this same machine.
6. Waterfall sustains ≥ 20 rows/s and the UI thread is never blocked > 100 ms
   during continuous decode (instrumented via the diagnostics panel).
7. Device unplug during `LISTENING` reaches `DEVICE_LOST` with a banner in
   ≤ 2 s, without crash or transcript loss; reconnection auto-resumes in
   ≤ 4 s.
8. Every unsupported-WAV class in §5.2 produces its specific error dialog and
   leaves the app state unchanged.
9. Every invalid-model class in §9 is rejected with the failing check named,
   and decoding continues on the prior/bundled model.
10. Sustained overload never crashes or emits stale text; the overrun counter
    matches injected drops in the overload test.
11. Cold start to interactive window ≤ 5 s from the packaged build on the
    reference machine.
12. Packaged builds launch and pass the golden-WAV check on Windows 11,
    Ubuntu 24.04 (PipeWire), and macOS 14 (with BlackHole installed).

## 12. macOS audio-routing guidance

Because macOS lacks native loopback, the app ships an in-app guide
(Help menu and the empty-device-selector call-to-action), rendered from
`resources/macos_loopback.md`:

1. Install a virtual audio driver — BlackHole 2ch (free, GPL-compatible
   distribution as a separate user-installed component; the app never bundles
   it) or Loopback.app (commercial).
2. Create a **Multi-Output Device** in Audio MIDI Setup containing both the
   speakers and BlackHole, so audio stays audible while being captured.
3. Set the Multi-Output Device as the system output.
4. In DeepFistTheApp, select **BlackHole 2ch** as the source.

The guide includes a troubleshooting section (no devices listed → driver not
installed; silence → system output not routed to the multi-output device) and
is linked from the macOS empty-selector state. The app detects the presence of
known virtual devices by name and deep-links the user to the relevant step.

## 13. Packaging & distribution

- **Python packaging:** `pyproject.toml` gains an `app` extra
  (`PySide6`, `onnxruntime`, `soundcard`) and a GUI entry point
  `deepfist-app = deepfistapp.main:main`; `deepfistapp*` joins the setuptools
  package find list. Developers run `pip install -e ".[app]"` then
  `deepfist-app`.
- **Binary builds:** PyInstaller one-dir bundles per OS (the bundled `.onnx`
  + sidecar + guide as data files): Windows zip + Inno Setup installer
  (mirroring Lyra's installer tooling), Linux tar.gz, macOS `.app` in a dmg
  (unsigned initially; Gatekeeper right-click-open documented in the README).
  **AppImage packaging is explicitly deferred beyond v1** — the tar.gz is the
  only v1 Linux artifact.
- **License compliance:** GPL-3.0-or-later for the app; About dialog and the
  bundle include the license text and third-party notices (PySide6 LGPL-3,
  onnxruntime MIT, numpy/scipy BSD, soundcard BSD) with the corresponding
  source-offer statement. PySide6 is used via its LGPL provisions —
  dynamically linked, unmodified, re-linkable, which the one-dir bundle
  layout preserves.
- The trained model remains out of git (existing `*.onnx` ignore rule);
  release artifacts are built by an explicit release script that exports the
  champion checkpoint at build time and records its hash in the About dialog.

## 14. Risks & mitigations

- **numpy/torch spectrogram drift** → parity + golden-text gates (§3.3,
  §10.3) make drift a hard test failure, with "depend on torch" as the
  documented fallback.
- **`soundcard` library gaps on a platform** → the `AudioSource` protocol
  isolates capture; a per-OS backend (e.g. `sounddevice`+WASAPI extras, or a
  small PyAudio WASAPI-loopback shim) can replace it behind the same
  interface without touching the engine or UI.
- **Lyra theme evolution** → tokens live in one module with a documented
  re-derivation source (`src/theme.h`); no scattered literals.
- **Model/preprocessing skew after future training** → the metadata sidecar
  check (§9) refuses mismatched models instead of decoding garbage.

---

*This document is the approved design of record for DeepFistTheApp. Any
implementation plan must be authored and approved separately.*
