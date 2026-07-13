"""Real-audio exact-callsign accuracy harness (fixed labeled eval set).

The reproducible metric for the train-to-95% loop: for each labeled real clip, decode with
exp15 + CTC rescorer and count an exact callsign match. Run with DEEPFIST_CONDITION=1.

The eval set is every clip under data/realset/ with a sidecar `.call` file (or a Lyra-shaped
session.json whose rbn.callsign / operator label is trusted), plus any listed in MANUAL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Operator-confirmed clips gathered live (ground truth = what WI9FD copied on the air).
MANUAL: list[tuple[str, str]] = [
    ("runs/manual_samples/2026-07-13_023146_N3BTM.wav", "N3BTM"),
    ("runs/manual_samples/2026-07-13_024038_W4MC.wav", "W4MC"),
    ("runs/rbn_harvest/2026-07-13_012734_7020kHz_W3RJ/audio.wav", "W3RJ"),
]


def labeled_clips() -> list[tuple[Path, str]]:
    """Collect (wav, truthCall) pairs: the MANUAL list plus anything dropped into
    data/realset/ (a <name>.wav beside a <name>.call text file)."""
    out: list[tuple[Path, str]] = []
    for rel, call in MANUAL:
        p = ROOT / rel
        if p.exists():
            out.append((p, call.upper()))
    rs = ROOT / "data" / "realset"
    if rs.is_dir():
        for wav in sorted(rs.glob("*.wav")):
            side = wav.with_suffix(".call")
            if side.exists():
                out.append((wav, side.read_text().strip().upper()))
    return out


def evaluate(ckpt: str) -> tuple[int, int, list]:
    import rbn_confirm
    clips = labeled_clips()
    rows, ok = [], 0
    for wav, truth in clips:
        picks = rbn_confirm.decode_recording(wav, ckpt)
        best = max(picks.items(), key=lambda kv: kv[1])[0] if picks else None
        hit = best == truth
        ok += hit
        rows.append((wav.name, truth, best, picks.get(truth), hit))
    return ok, len(clips), rows


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/exp15/model.pt"
    ok, n, rows = evaluate(ckpt)
    print(f"eval set: {n} labeled real clips   model: {ckpt}\n")
    print(f"  {'clip':40} {'truth':8} {'decoded':8} {'truth_margin':>12} {'hit':>4}")
    for name, truth, best, tm, hit in rows:
        print(f"  {name[:40]:40} {truth:8} {str(best or '-'):8} "
              f"{('' if tm is None else f'{tm:.1f}'):>12} {'YES' if hit else '':>4}")
    acc = ok / n if n else 0.0
    print(f"\nexact-callsign accuracy: {ok}/{n} = {acc:.1%}")


if __name__ == "__main__":
    main()
