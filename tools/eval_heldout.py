"""Leakage-safe held-out real eval: the 3 operator clips + the held-out anchors listed in
data/realset_holdout.txt (never used in real-blend training). Exact-callsign via rescorer.
Run one model per process (rbn_confirm.get_rescorer caches globally). DEEPFIST_CONDITION=1.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

OPERATOR = [
    ("runs/manual_samples/2026-07-13_023146_N3BTM.wav", "N3BTM"),
    ("runs/manual_samples/2026-07-13_024038_W4MC.wav", "W4MC"),
    ("runs/rbn_harvest/2026-07-13_012734_7020kHz_W3RJ/audio.wav", "W3RJ"),
]


def heldout_clips():
    clips = []
    for rel, call in OPERATOR:
        p = ROOT / rel
        if p.exists():
            clips.append((p, call))
    hf = ROOT / "data" / "realset_holdout.txt"
    ho = set(hf.read_text().split()) if hf.exists() else set()
    for sj in sorted((ROOT / "runs" / "rbn_anchors").glob("*/session.json")):
        if sj.parent.name in ho:
            m = json.loads(sj.read_text())
            clips.append((sj.parent / m["audio"][0]["file"], m["rbn"]["callsign"].upper()))
    return clips


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
    import rbn_confirm as R
    clips = heldout_clips()
    ok, rows = 0, []
    for wav, call in clips:
        picks = R.decode_recording(wav, ck)
        best = max(picks.items(), key=lambda kv: kv[1])[0] if picks else None
        hit = best == call
        ok += hit
        rows.append((call, best, hit))
    print(f"{ck}: held-out real {ok}/{len(clips)} = {ok/len(clips):.0%}")
    for c, b, h in rows:
        print(f"   {c:8} -> {str(b or '-'):8} {'OK' if h else ''}")


if __name__ == "__main__":
    main()
