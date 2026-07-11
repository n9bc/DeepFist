"""Generate one CW clip, write it to a .wav, and print its label/meta."""
import argparse
import wave
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.synth.generator import generate, GenConfig


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="clip.wav")
    ap.add_argument("--window", type=float, default=6.0)
    args = ap.parse_args()
    s = generate(seed=args.seed, config=GenConfig(window_s=args.window))
    write_wav(args.out, s.audio, s.meta["sample_rate"])
    print(f"label: {s.label!r}")
    print(f"meta:  {s.meta}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
