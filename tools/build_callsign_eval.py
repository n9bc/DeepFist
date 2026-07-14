"""Build a leakage-safe held-out callsign eval from the RBN-labeled anchor pool.

The old eval (data/realset_holdout.txt, 10 clips) shares ALL its callsigns with the
training clips in data/realset_train/ -> callsign/operator leakage + only 13 questions.
This rebuilds a bigger, callsign-DISJOINT held-out set and a disjoint training pool.

Leakage rule: a callsign trained on (data/realset_train/<CALL>_*.wav) is banned from eval.
Independence: hold out by CALLSIGN (one clip per distinct station), and guarantee the eval
callsigns and the training-pool callsigns never overlap (no shared operator fist).

Emits:
  data/callsign_eval_heldout.txt  - anchor dir names, FROZEN eval (never train on these)
  data/callsign_train_pool.txt    - anchor dir names, callsign-disjoint training pool
"""
from __future__ import annotations
import json
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_TARGET = 100  # distinct-callsign clips to freeze as the held-out eval


def leaked_callsigns() -> set[str]:
    calls = set()
    for w in glob.glob(str(ROOT / "data" / "realset_train" / "*.wav")):
        base = os.path.basename(w)
        # <CALL>_<nnn>.wav
        calls.add(base.rsplit("_", 1)[0].upper())
    return calls


def load_pool():
    """Return list of dicts: dir_name, callsign, agree, snr, wpm, wav (abs)."""
    rows = []
    for d in ("rbn_anchors", "rbn_harvest"):
        for sj in glob.glob(str(ROOT / "runs" / d / "*" / "session.json")):
            m = json.load(open(sj))
            call = (m.get("rbn", {}) or {}).get("callsign")
            if not call:
                continue
            p = Path(sj).parent
            wav = p / (m["audio"][0]["file"] if m.get("audio") else "audio.wav")
            if not wav.exists():
                continue
            rows.append({
                "dir": p.name, "call": call.upper(),
                "agree": (m.get("decode", {}) or {}).get("agree"),
                "snr": (m.get("rbn", {}) or {}).get("peak_snr"),
                "wpm": (m.get("rbn", {}) or {}).get("wpm"),
                "wav": str(wav),
            })
    return rows


def main():
    leaked = leaked_callsigns()
    pool = load_pool()

    # group clean (non-leaked) clips by callsign
    by_call: dict[str, list[dict]] = {}
    for r in pool:
        if r["call"] in leaked:
            continue
        by_call.setdefault(r["call"], []).append(r)

    # deterministic: sort callsigns; for each pick the highest-SNR clip as representative
    clean_calls = sorted(by_call)
    for c in clean_calls:
        by_call[c].sort(key=lambda r: (-(r["snr"] or -1), r["dir"]))

    eval_calls = clean_calls[:EVAL_TARGET]
    train_calls = clean_calls[EVAL_TARGET:]

    eval_rows = [by_call[c][0] for c in eval_calls]                 # one clip per eval call
    train_rows = [r for c in train_calls for r in by_call[c]]        # all clips of train calls

    (ROOT / "data" / "callsign_eval_heldout.txt").write_text(
        "\n".join(r["dir"] for r in eval_rows) + "\n")
    (ROOT / "data" / "callsign_train_pool.txt").write_text(
        "\n".join(r["dir"] for r in train_rows) + "\n")

    def hard(rs):
        return sum(1 for r in rs if r["agree"] is False)

    print(f"pool: {len(pool)} RBN-labeled clips, {len(set(r['call'] for r in pool))} distinct calls")
    print(f"leaked (in training, banned from eval): {len(leaked)} calls")
    print(f"clean distinct calls available: {len(clean_calls)}")
    print()
    print(f"EVAL  (frozen): {len(eval_rows)} clips / {len(eval_calls)} distinct calls "
          f"-> {hard(eval_rows)} model currently MISSES, {len(eval_rows)-hard(eval_rows)} it gets")
    print(f"TRAIN pool     : {len(train_rows)} clips / {len(train_calls)} distinct calls "
          f"(callsign-disjoint from eval)")
    print()
    print("eval sample (call, snr, model-currently-right?):")
    for r in eval_rows[:12]:
        print(f"   {r['call']:9} {str(r['snr'])+'dB':>6}  {'HIT' if r['agree'] else 'miss'}")
    print(f"\nwrote data/callsign_eval_heldout.txt  ({len(eval_rows)})")
    print(f"wrote data/callsign_train_pool.txt    ({len(train_rows)})")


if __name__ == "__main__":
    main()
