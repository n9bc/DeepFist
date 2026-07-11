"""Evaluate a checkpoint (or fresh model) and print the per-SNR CER table."""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.model.net import CwCtcNet
from deepfist.train.metrics import evaluate_per_snr


def run_eval(model, snr_points, clips_per_point, device="cpu"):
    model = model.to(device)
    return evaluate_per_snr(model, snr_points, clips_per_point, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--downsample", type=int, default=2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CwCtcNet(time_downsample=args.downsample)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=device))
    table = run_eval(net, [10, 6, 3, 0, -3, -6], args.clips, device=device)
    for k, v in table.items():
        print(f"{k:+.0f} dB : CER {v:.3f}")


if __name__ == "__main__":
    main()
