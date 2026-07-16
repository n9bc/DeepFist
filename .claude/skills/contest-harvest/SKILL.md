---
name: contest-harvest
description: Use when harvesting real off-air CW during a live contest (CWOps CWT, etc.) through the operator's Lyra receiver + the Reverse Beacon Network, to build RBN-labeled DeepFist training data. Covers pre-flight safety, the hands-off capture loop, dataset build, and the gated retrain.
---

# RBN-confirmed contest harvest (DeepFist)

Harvest real, strong, off-air CW during a contest and label it with RBN callsign
consensus, so DeepFist trains on the true deployment distribution (the operator's
own receiver). The operator tunes; we follow, capture, and label. **We never move
the dial and never touch the deployed model.**

## THE most important rule: a contest window is ~1 hour. Run HANDS-OFF.

The one thing that wrecked the first harvest: stopping mid-window to ask the
operator per-clip verification questions. **Don't.** Once the capture loop is
running:
- Do the sideband/pitch sanity check exactly **once**, silently, on the first
  clip. If RBN matches a real strong call at the operator's dial, the mapping is
  right — say so in one line and stop checking.
- After that, relay each capture in **one terse line** (RBN call, keyed y/n, read).
  **Ask nothing.** The operator drives; you follow.
- Only interrupt if captures are *systematically* broken (every clip not-keyed, or
  RBN never matches → likely wrong sideband; flip CWU↔CWL and continue).

## Pre-flight (do this BEFORE the window opens, not during)

1. **Repo safety** — other sessions do heavy work here. Before running anything:
   `git -C c:/dev/DeepFist status --short` and (if relevant) `git -C c:/dev/diddle status --short`.
   Note uncommitted model swaps / WIP so you don't clobber them. Write only to NEW
   `runs/` dirs; never overwrite an existing `expN`.
2. **Find the current champion** — do NOT assume exp15/exp16/exp27. Grep HANDOFF.md
   for the latest `CHAMPION` / `ADOPTED` section; use that ckpt for the live decode
   and as the warm-start base. Confirm which model Lyra actually runs.
3. **TCI up?** `bash -c '(echo>/dev/tcp/127.0.0.1/40001)>/dev/null 2>&1 && echo OPEN || echo CLOSED'`
   (Lyra=40001, Thetis=50001). websockets must be installed in `.venv`.
4. **Operator callsign** for the RBN telnet login (e.g. N9BC). **Band + sideband:**
   20m→CWU, 40m/80m→CWL. Pitch default 650 Hz (tolerance absorbs ±few-hundred Hz).
5. Always run captures/decode with `DEEPFIST_CONDITION=1` (the deployed model is conditioned).

## Capture loop

Start the passive RBN spot logger (no radio control — just records who's where):

```
.venv/Scripts/python.exe tools/rbn_spot_logger.py --call <CALL> --out runs/rbn_spots_live.jsonl
```

Then the hands-free follow-VFO capturer (operator tunes → each settled new freq
auto-records `--dwell`s, matched to RBN, decoded by the champion):

```
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/follow_vfo_capture.py \
    --dwell 10 --sideband CWU --ckpt runs/<champion>/model.pt --out runs/rbn_cwt_op
```

Launch both **without** a trailing `&` inside a background call (the `&` orphans the
python and the wrapper reports a bogus exit). Monitor the log for events:
`\[tune\]|\[save\]|ERROR|Traceback`. Relay `[save]` lines tersely.

**Higher yield (PREFERRED): continuous wide-band recording.** Station-by-station
tops out ~1 clip/30s. Instead, park the dial at the low edge of the activity and
record a wide slice of the segment continuously, then align RBN spots offline and
cut one clean clip per station — dozens of clips per sitting (HANDOFF §18.22). Built
and DSP-validated:

```
# spot logger already running; park Lyra at the low edge (e.g. 14025), then:
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/band_record.py \
    --minutes 12 --width 5000 --sideband CWU --out runs/band
# after: extract labeled single-signal clips (isolate each RBN offset -> 600 Hz)
DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/band_extract.py \
    --rec runs/band/<stamp>_band --spots runs/rbn_spots_live.jsonl \
    --out runs/real_band_train --require-agree
```

`--width 5000` covers ~5 kHz (several packed contest stations). **Measured: Lyra caps
the CW filter at ~6 kHz** — requesting more just clamps there and wastes extraction on
an empty 6+ kHz range, so keep width ≤5000. Verify the achieved passband on a 30s test
capture (`--minutes 0.5`) and look for the noise-floor cliff.
`--require-agree` keeps only clips whose DeepCW decode contains the RBN call (proves
correct isolation). `follow_vfo_capture.py` remains the operator-tuned fallback.

## After the window: build + gated retrain

1. **Build dataset** — teacher-label the saved clips into a WMR set:
   `tools/build_real_dataset.py --wav runs/rbn_cwt_op/*/audio.wav --out runs/real_<tag>_train`
   (drops not-keyed / silent windows). Hold out a few whole sessions for eval.
2. **Train** exp N+1: warm-start from the **champion**, blend the new real set
   (`--wmr ... --wmr-prob ~0.4`), `--qrm-prob 0` (single-signal), `DEEPFIST_CONDITION=1`,
   to a NEW `runs/expN+1` dir.
3. **Gate before ANY adoption** (this is the guardrail exp25 failed):
   - real ARRL CER no-regression (`tools/eval_real_session.py`, champion vs new),
   - preserve exp27's BT/prosign fix (`scratch_bt_test.py`),
   - improve held-out real / operator clips.
   Adopt (export → Lyra `models/deepfist.onnx`, back up the old one) **only** with the
   operator's explicit go-ahead. Otherwise keep the champion.

## Tools (all in tools/)

- `rbn_spot_logger.py` — passive RBN telnet → timestamped spot JSONL. No radio control.
- `band_record.py` — PREFERRED: widen the CW filter, record a wide slice continuously.
- `band_extract.py` — offline: RBN-align + heterodyne-isolate each station → labeled WMR clips.
- `follow_vfo_capture.py` — poll VFO; on each settled new freq, record + RBN-match + decode + save.
- `capture_here.py` — one-shot capture of the current dial (turn-based alternative).
- `rbn_harvest.py` — the *reactive* auto-chaser (drives the VFO). Has known drift
  (§18.22) — prefer the operator-driven tools above during a contest.
- `rbn_confirm.py` / `eval_real_session.py` — consume the saved Lyra-shaped clips.

## Known pitfalls

- **Drift** (§18.22): reactive chasing records the wrong station after the CQ ends.
  Operator-driven capture avoids it (signal present by construction).
- **`[NOT keyed-CW]`** = the 10s window caught a between-callers gap. Normal; those
  clips are filtered at build time. Not a bug.
- **Wrong sideband** = RBN estimate off ~1.3 kHz → matches nothing. Symptom: every
  clip "no match". Fix: flip `--sideband`.
- Session clips are saved Lyra-shaped (`session.json` + `audio.wav`, `rbn.callsign` =
  label) so the existing eval/build tools consume them unchanged.
