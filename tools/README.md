# DeepFist tools & real-data workflow

Reference for the tooling built during the 2026-07-12 real-data work (HANDOFF §17).
Run everything with the venv Python: `.venv/Scripts/python.exe`. Datasets/artifacts
live under `runs/` (gitignored). Set `OMP_NUM_THREADS=1` for training.

## The conditioner (read this first)

The live decoder (diddle `cw_neural.rs`) **conditions** audio before the model:
AGC → tone AFC → 90 Hz matched bandpass → recenter to 600 Hz → peak-norm. Training and
eval must match, so conditioning is gated by an env var:

```
DEEPFIST_CONDITION=1   # apply the conditioner in loaders + eval tools
```

`deepfist/features/conditioner.py` is the vectorized twin of the Rust conditioner
(`condition()`, `maybe_condition()`). **A model trained with `DEEPFIST_CONDITION=1`
MUST be evaluated/deployed with conditioning** (diddle already conditions). Consequence:
DeepFist is a **single-signal decoder** — train on single-signal audio (`--qrm-prob 0`);
multi-signal contest evals (wmr_evalA) are not meaningful for a conditioned model.

## Tools

| Tool | Purpose |
|---|---|
| `scripts/tci_capture.py` | Capture live TCI RX audio (Lyra/Thetis) → WAV for offline decode. |
| `tools/teacher_label.py` | Decode one clip with DeepCW (teacher) vs a DeepFist ckpt — sanity/compare. |
| `tools/build_real_dataset.py` | Slide a window over real audio, DeepCW-label each, tokenize + peak-norm → WMR-format dataset. `--end` for leakage-free splits, `--agree` consensus filter. |
| `tools/build_arrl_hispeed.py` | Orchestrate ARRL train/eval sets across speeds (edit `SPEEDS`), leakage-free tail holdout. |
| `tools/eval_real_session.py` | **Honest real eval:** windowed full-session decode vs a ground-truth `.txt` transcript, space-normalized CER, per model. |
| `tools/benchmark_vs_deepcw.py` | CER benchmark of DeepFist ckpt(s) vs DeepCW on a WMR-format dir (per-SNR, `--latency`). |
| `tools/rescore.py` | **CTC hypothesis rescorer** (WSJT-X "Deep Search" for CW): score candidate callsigns against the model's CTC lattice via `F.ctc_loss`, instead of text edit distance (`scp_correct.py`). Default mode swaps SCP candidates into the greedy decode's callsign words; `--probe CALL` (+ `--t0/--t1` slice) scores explicit candidates on garbled clips. Decode-time only. **Ported live**: diddle `dsp/rescore.rs` runs it in `CwEngine` per decode window and emits `cw:calls` (offline check: `DIDDLE_SCP=... cargo run --release --example cw_decode_wav -- clip.wav`). |
| `tools/scp_correct.py` | Text-only SCP snap (edit distance ≤1, unique match) — the simpler cousin of `rescore.py`; its logic is what's ported in diddle's `ScpDb::correct`. |
| `tools/rbn_confirm.py` | **Independent callsign ground truth** via the Reverse Beacon Network. Cross-references a Lyra recording (session.json → dial freq, sideband, UTC, duration) against the RBN daily spot archive (`data.reversebeacon.net/rbn_history/YYYYMMDD.zip`), ranking real calls that skimmers spotted in that freq+time slice by skimmer-agreement count. Breaks the DeepCW teacher ceiling for callsign copy. `--selftest` validates the matcher; archive publishes only after a UTC day closes (same-day clips wait ~1 day). |
| `scripts/train.py --init CKPT` | Warm-start (fine-tune) from a checkpoint instead of from scratch. |

DeepCW (teacher/reference) is loaded from `C:\dev\deepcw-engine` at runtime — AGPL,
dev-only, never shipped/committed (see `benchmark_vs_deepcw.py` header).

## End-to-end: real-data fine-tune (what produced exp14/exp15)

```bash
# 1. Get real audio with ground-truth labels: ARRL W1AW code practice
#    https://www.arrl.org/code-practice-files  (MP3 per speed + .txt transcript)
curl -sSL -A "Mozilla/5.0" -o runs/real/arrl/arrl_25wpm.mp3 \
     "https://www.arrl.org/files/file/Morse/260303_25WPM.mp3"
ffmpeg -y -i runs/real/arrl/arrl_25wpm.mp3 -ac 1 -ar 48000 runs/real/arrl/arrl_25wpm_mono.wav
#    (YouTube sources need yt-dlp: `.venv/Scripts/python.exe -m yt_dlp -x --audio-format wav ...`)

# 2. Build teacher-labeled train/eval sets (DeepCW labels each window)
.venv/Scripts/python.exe tools/build_arrl_hispeed.py         # -> runs/real_arrl_train, runs/real_arrl_eval

# 3. Fine-tune, CONDITIONED, single-signal synthetic blend, warm-started
DEEPFIST_CONDITION=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe scripts/train.py \
    --init runs/exp11/model.pt --wmr runs/real_arrl_train --wmr-prob 0.4 \
    --width 2.5 --steps 4000 --lr 1e-4 --snr-min -12 --snr-max 10 --qrm-prob 0.0 \
    --out runs/exp15

# 4. Evaluate (always with DEEPFIST_CONDITION=1 for conditioned models)
#    a) ground-truth ARRL sweep vs DeepCW:
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/eval_real_session.py \
    --wav runs/real/arrl/arrl_25wpm_mono.wav --txt runs/real/arrl/arrl_25wpm.txt \
    --ckpt runs/exp15/model.pt --deepcw --win 15 --norm-peak
#    b) held-out real (teacher labels):
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/benchmark_vs_deepcw.py \
    --eval-dir runs/real_arrl_eval --ckpt runs/exp15/model.pt

# 5. Export + deploy (ONNX is spectrogram-in; diddle conditions before it)
.venv/Scripts/python.exe scripts/export.py --ckpt runs/exp15/model.pt --out runs/deepfist.onnx
#    copy runs/deepfist.onnx + .json into diddle's model dir (see below)
```

## Live on-air test (Lyra/Thetis)

```bash
# grab a few seconds of the tuned signal
.venv/Scripts/python.exe scripts/tci_capture.py --uri ws://127.0.0.1:40001 --seconds 18 --out runs/lyra_live.wav
# decode/compare through the conditioned pipeline
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/teacher_label.py --wav runs/lyra_live.wav --compare runs/exp15/model.pt
```

Diddle A/B: models staged in `C:\dev\diddle\models_ab\{exp14,exp15}\deepfist.onnx`; select
one live with `$env:DEEPFIST_MODEL_DIR="...\models_ab\exp15"` before `npm run tauri dev`.
