# DeepFist

A clean-room, self-trained neural network **CW (Morse code) decoder** — and a
live, on-screen **copy display** that reads real off-air CW like a CW reader.

DeepFist decodes Morse the way modern speech recognition works:
`audio → conditioning → spectrogram → CNN + CTC → text`. Instead of hand-written
timing rules, it is trained on synthetically generated CW degraded with noise,
fading (QSB), and interference, then adapted on real off-air recordings — so it
stays readable on weak, messy, real-world signals where threshold decoders fall
apart.

**Hard rule:** no signal energy → no characters. The decoder stays silent on an
empty frequency instead of hallucinating text.

## Status

**Working.** The model trains, evaluates, exports to ONNX, and runs live off a
radio over TCI. On real ARRL code-practice audio the champion model copies plain
text at roughly **6–12% character error rate across 10–30 WPM** (measured full
session, space-normalized). Clean copy of arbitrary *hand-sent* fists is the open
frontier — see [`HANDOFF.md`](HANDOFF.md).

> Trained weights are **not** committed (the `runs/` directory and `*.pt` files
> are git-ignored to keep the repo light). You train your own model, or obtain a
> checkpoint separately, then point the tools at it. See [Training](#training).

## Install

Requires **Python 3.11+**.

```bash
git clone <this-repo> && cd DeepFist
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -e .
```

`torch` installs the default (CPU) build. For an NVIDIA GPU, install the CUDA
build first, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Runtime dependencies: `numpy`, `scipy`, `torch`, `websockets` (TCI). The
soundcard-based live decoder additionally needs `sounddevice`
(`pip install -e ".[audio]"`). Dev/test extras: `pip install -e ".[dev]"`.

## Quick start

### Live decode from a radio (TCI) — the flagship

Streams RX audio from an SDR that speaks TCI (Lyra / ExpertSDR3 on
`ws://127.0.0.1:40001`, Thetis on `:50001`), and streams decoded text to the
terminal as it settles (~1 s behind live).

```bash
python scripts/tci_decode.py --ckpt path/to/model.pt
```

What it does automatically:
- **Squelch** — an in-band *keying* detector (`tools/squelch.py`). Dead air and
  the receiver's AGC tone score low; only genuinely keyed CW opens the gate, so
  no characters are printed on an empty frequency. Tune with `--squelch <n>`
  (default 12; higher = stricter).
- **De-spike** — an impulse-noise blanker (`tools/despike.py`) that removes
  static crashes before decoding. On by default; `--no-despike` to disable.
- **Conditioning** — AGC → tone-lock → narrow band-pass → re-center to 600 Hz,
  the front-end the model was trained with (mandatory for good copy).

Useful flags: `--uri`, `--rx`, `--window` (context seconds, default 6),
`--tick` (decode interval), `--guard` (commit latency), `--seconds` (auto-stop).

### Decode / evaluate a recording against ground truth

`tools/eval_real_session.py` decodes a WAV in windows and reports character error
rate vs a transcript. Set `DEEPFIST_CONDITION=1` so the conditioning front-end is
applied (required to reproduce the model's real-audio accuracy).

```bash
DEEPFIST_CONDITION=1 python tools/eval_real_session.py \
    --wav recording.wav --txt transcript.txt --ckpt path/to/model.pt
```

Optional: `--despike` (impulse blank each window), `--tempo <wpm>` (speed-warp;
off by default and generally not recommended — see HANDOFF §18.24), `--norm-peak`.

### Capture off-air audio from the radio

```bash
python scripts/tci_capture.py --seconds 30 --out capture.wav
```

### Live decode from a soundcard (no TCI)

For a rig feeding audio to a virtual/real input device (needs `sounddevice`):

```bash
python scripts/live_decode.py --device "Virtual Audio Cable" --ckpt path/to/model.pt
```

## How it works

```
radio audio (48 kHz)
   |  decimate -> 3200 Hz
   |--> squelch (keying gate, on RAW audio) --- no signal? -> print nothing
   v
 de-spike -> conditioner (AGC / tone-lock / band-pass / re-center 600 Hz)
   |
   v
 spectrogram -> CwCtcNet (CNN + temporal conv, CTC head)
   |
   v
 greedy CTC decode -> streamed text
```

The model is a compact convolutional CTC network (`deepfist/model/net.py`,
`CwCtcNet`) operating at 3200 Hz. The squelch runs on *raw* audio (conditioning
would fabricate a tone from noise); conditioning and de-spike run only once a
window has passed the gate.

## Training

Training generates synthetic CW on the fly (no dataset download) and can blend in
real off-air clips. Basic run:

```bash
python scripts/train.py --out runs/myexp --steps 4000 --width 2.5
```

Key knobs: `--width` (model size), `--snr-min/--snr-max`, `--qrm-prob`,
`--wmr <dir> --wmr-prob` (blend a real/WMR clip dataset), `--init <ckpt>`
(warm-start / fine-tune), `--lr`. The champion recipe (warm-start + real-audio
blend, and the hard-won rules about what regresses real accuracy) is documented
in [`HANDOFF.md`](HANDOFF.md) and [`CLAUDE.md`](CLAUDE.md).

- Evaluate synthetic per-SNR: `python scripts/evaluate.py --ckpt runs/myexp/model.pt`
- Export to ONNX: `python scripts/export.py --ckpt runs/myexp/model.pt --out deepfist.onnx`
- Run the test suite: `pytest`

## Project layout

```
deepfist/        core library
  synth/           synthetic CW generator + channel model
  features/        spectrogram + conditioner (front-end)
  model/           CwCtcNet + CTC decode
  morse/           alphabet / token tables
  train/           training loop + metrics
  export/          ONNX export
scripts/         entry points: train, generate, evaluate, export,
                 tci_decode (live TCI), tci_capture, live_decode (soundcard)
tools/           squelch, despike, tempo, cw_lm, eval + analysis utilities
tests/           pytest suite
```

## Relationship to DeepCW

DeepFist is **inspired by** e04's excellent
[DeepCW](https://github.com/e04/deepcw-engine), which proved a neural CW decoder
can beat traditional decoders in noise. DeepCW is licensed **AGPL-3.0-only**,
which is why DeepFist is a fresh, independent implementation: DeepCW is used only
as a *conceptual reference*, never copying its code or trained weights.

## License

**MIT** — see [`LICENSE`](LICENSE). Permissive by design, so it can be embedded in
other applications (e.g. [SDRLoggerPlus](https://github.com/N8SDR1/SDRLoggerPlus)).
