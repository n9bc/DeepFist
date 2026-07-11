"""Launch a training run."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.train.loop import train, TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch, num_workers=args.workers,
                      time_downsample=args.downsample, out_dir=args.out)
    train(cfg)


if __name__ == "__main__":
    main()
