"""Super Check Partial callsign corrector — snap a garbled decoded call to the
nearest real MASTER.SCP entry. Designed to be real-time: O(1) exact-match, and
fuzzy match only against same-length(+/-1) buckets so it's sub-millisecond.

Reference implementation (Python) for porting into the Rust/C++ decode layer.
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

CALL_RE = re.compile(r"^[A-Z0-9]{0,3}?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}$")  # liberal call shape


def load_scp(path: str):
    calls = set()
    for line in Path(path).read_text().splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        calls.add(s)
    by_len = defaultdict(list)
    for c in calls:
        by_len[len(c)].append(c)
    return calls, by_len


def _within1(a: str, b: str) -> bool:
    """True if edit distance(a, b) <= 1 (sub / ins / del). Fast, length-aware."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    # one insertion/deletion: make a the longer
    if la < lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        elif skipped:
            return False
        else:
            skipped = True; i += 1
    return True


def correct_call(token: str, calls: set, by_len: dict, max_edits: int = 1):
    """Return (corrected_call, matched). Exact hit -> itself. Else the UNIQUE
    within-`max_edits` SCP call, or the original if none/ambiguous."""
    t = token.upper()
    if t in calls:
        return t, True
    cands = []
    for L in (len(t) - 1, len(t), len(t) + 1):
        for c in by_len.get(L, ()):  # only same-ish length bucket
            if _within1(t, c):
                cands.append(c)
                if len(cands) > 1:
                    return token, False  # ambiguous -> don't guess
    if len(cands) == 1:
        return cands[0], True
    return token, False


def is_call(tok: str) -> bool:
    return len(tok) >= 3 and CALL_RE.match(tok) is not None


if __name__ == "__main__":
    scp = sys.argv[1] if len(sys.argv) > 1 else "data/MASTER.SCP"
    calls, by_len = load_scp(scp)
    print(f"loaded {len(calls)} calls\n")

    # tokens seen in the real decodes + a couple of realistic 1-off garbles
    tests = ["W33G", "WC3", "K7CQ", "N8SDR", "WQ6X", "WQ6Y", "K3LR", "K3LS", "AA3B", "ZZ9ZZ"]
    t0 = time.perf_counter()
    N = 2000
    for _ in range(N):
        for tk in tests:
            correct_call(tk, calls, by_len)
    dt = (time.perf_counter() - t0) / (N * len(tests)) * 1e6
    print(f"{'token':10} -> {'corrected':10} match   (avg {dt:.1f} us/lookup)\n" + "-" * 40)
    for tk in tests:
        corr, ok = correct_call(tk, calls, by_len)
        flag = "SNAP" if ok and corr != tk else ("ok" if ok else "-")
        print(f"{tk:10} -> {corr:10} {flag}")
