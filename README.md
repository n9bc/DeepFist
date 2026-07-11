# DeepFist

A clean-room, self-trained neural network **CW (Morse code) decoder**.

DeepFist learns to decode Morse from audio the way modern speech recognition works:
`audio → spectrogram → CNN + CTC → text`. Instead of hand-written timing rules, it is
trained on synthetically generated CW that is deliberately degraded with noise, fading
(QSB), and interference — so it stays accurate on weak, messy, real-world signals.

## Status

**Pre-implementation / design phase.** No model or training code has been written yet.
See [`HANDOFF.md`](HANDOFF.md) for the full plan, decisions, and where to resume.

## Goals

- A single-signal, real-time CW decoder good in low-SNR conditions.
- 100% original code and a model trained on our own synthetic data — no third-party
  weights or copyleft code — so it can ship inside other applications freely.
- Exported as a portable **ONNX** model plus a small inference library, so
  [SDRLoggerPlus](https://github.com/N8SDR1/SDRLoggerPlus) (and anything else) can
  consume it when it's ready.

## Relationship to DeepCW

DeepFist is **inspired by** e04's excellent [DeepCW](https://github.com/e04/deepcw-engine)
project, which proved a neural CW decoder can beat traditional decoders in noise.
DeepCW is licensed **AGPL-3.0-only**, which is why DeepFist is a fresh, independent
implementation: we use DeepCW only as a *conceptual reference*, never copying its code
or trained weights. See [`HANDOFF.md`](HANDOFF.md#licensing) for details.

## License

TBD — intended to be permissive (e.g. MIT) so SDRLoggerPlus and others can embed it.
