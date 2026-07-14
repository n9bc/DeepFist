"""Score a checkpoint on the leakage-safe frozen callsign eval (data/callsign_eval_heldout.txt,
built by tools/build_callsign_eval.py — 100 distinct stations, callsign-disjoint from training).

Run ONE model per process (rbn_confirm.get_rescorer caches the net globally).
  DEEPFIST_CONDITION=1 .venv/Scripts/python.exe tools/eval_callsign_heldout.py runs/exp16/model.pt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def clips():
    names = (ROOT / "data" / "callsign_eval_heldout.txt").read_text().split()
    out = []
    for n in names:
        for d in ("rbn_anchors", "rbn_harvest"):
            sj = ROOT / "runs" / d / n / "session.json"
            if sj.exists():
                m = json.loads(sj.read_text())
                wav = sj.parent / (m["audio"][0]["file"] if m.get("audio") else "audio.wav")
                out.append((wav, m["rbn"]["callsign"].upper()))
                break
    return out


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"
    import rbn_confirm as R
    cl = clips()
    ok, rows = 0, []
    for wav, call in cl:
        picks = R.decode_recording(wav, ck)
        best = max(picks.items(), key=lambda kv: kv[1])[0] if picks else None
        hit = best == call
        ok += hit
        rows.append((call, best, hit))
    print(f"{ck}: leakage-safe callsign {ok}/{len(cl)} = {ok/len(cl):.0%}")
    misses = [(c, b) for c, b, h in rows if not h]
    print(f"  hits: {ok}   misses: {len(misses)}")
    for c, b in misses[:15]:
        print(f"    {c:9} -> {b or '-'}")


if __name__ == "__main__":
    main()
