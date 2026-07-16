"""Error decomposition of a checkpoint on real ARRL audio (ground truth), all speeds.

For each WPM: CER, and a breakdown of the edit distance into substitutions /
insertions / deletions, the top confusions, and a word-count (spacing) check.
Reveals *what kind* of errors dominate — the analysis that found exp16's
insertion/mis-segmentation failure mode (HANDOFF §18.25).

  .venv/Scripts/python.exe tools/error_analysis.py [runs/<exp>/model.pt]

Baselines (exp16): CER 11.7/8.4/9.7/5.5/5.9 @ 10/15/20/25/30 wpm; ins ~75%, del ~0%.
"""
import os, sys
os.environ["DEEPFIST_CONDITION"] = "1"          # model was trained conditioned
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
from collections import Counter
import numpy as np
import importlib.util as ilu

# import eval_real_session helpers
spec = ilu.spec_from_file_location("ers", str(ROOT / "tools" / "eval_real_session.py"))
E = ilu.module_from_spec(spec); spec.loader.exec_module(E)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/exp16/model.pt"


def align(ref, hyp):
    """Levenshtein backtrace over char lists -> (match, sub, ins, del, confusions)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + c)
    i, j = n, m
    mt = sb = ins = de = 0
    conf, insc, delc = Counter(), Counter(), Counter()
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            if ref[i-1] == hyp[j-1]:
                mt += 1
            else:
                sb += 1; conf[(ref[i-1], hyp[j-1])] += 1
            i -= 1; j -= 1
        elif j > 0 and d[i, j] == d[i, j-1] + 1:
            ins += 1; insc[hyp[j-1]] += 1; j -= 1
        else:
            de += 1; delc[ref[i-1]] += 1; i -= 1
    return mt, sb, ins, de, conf, insc, delc


for w in (10, 15, 20, 25, 30):
    wav = f"runs/real/arrl/arrl_{w}wpm_mono.wav"
    txt = f"runs/real/arrl/arrl_{w}wpm.txt"
    ref_sp = E.keep_tokenizable(E.clean_transcript(Path(txt).read_text(encoding="utf-8", errors="ignore")))
    sr, a = E.load_wav(wav, False)
    pred_sp = E.decode_ours(CKPT, a, sr, 15.0, 15.0)
    ref = ref_sp.replace(" ", ""); hyp = pred_sp.replace(" ", "")
    mt, sb, ins, de, conf, insc, delc = align(list(ref), list(hyp))
    tot = sb + ins + de
    cer = tot / max(1, len(ref))
    # spacing: word-count delta
    rw, hw = len(ref_sp.split()), len(pred_sp.split())
    print(f"\n=== ARRL {w} WPM === CER {cer*100:5.1f}%  (ref {len(ref)} chars, {rw} words; pred {hw} words)")
    print(f"  errors: sub {sb} ({sb/max(1,tot)*100:.0f}%)  ins {ins} ({ins/max(1,tot)*100:.0f}%)  del {de} ({de/max(1,tot)*100:.0f}%)")
    print(f"  top confusions (ref->pred): " + ", ".join(f"{r}->{p}:{c}" for (r, p), c in conf.most_common(8)))
    print(f"  top inserted: " + ", ".join(f"{repr(ch)}:{c}" for ch, c in insc.most_common(6)))
    print(f"  top deleted:  " + ", ".join(f"{repr(ch)}:{c}" for ch, c in delc.most_common(6)))
