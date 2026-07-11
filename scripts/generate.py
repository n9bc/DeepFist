"""Print N generated CW labels + meta for inspection / smoke testing."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.synth.generator import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for i in range(args.n):
        s = generate(seed=args.seed + i)
        print(f"[{i}] {s.label!r}  "
              f"wpm={s.meta['wpm']:.1f} snr={s.meta['snr_db']:.1f} "
              f"pitch={s.meta['pitch_hz']:.0f}")


if __name__ == "__main__":
    main()
