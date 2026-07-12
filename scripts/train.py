"""Launch a training run."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.train.loop import train, TrainConfig
from deepfist.synth.generator import GenConfig
from deepfist.synth.channel import ChannelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--wmr", default="", help="WebMorseRunner dataset dir to blend in")
    ap.add_argument("--wmr-prob", type=float, default=0.5, dest="wmr_prob")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=float, default=1.0, help="channel-width multiplier (1.0=~528k params)")
    # non-WMR synth difficulty knobs (widen low end to harden low-SNR robustness)
    ap.add_argument("--snr-min", type=float, default=-6.0, dest="snr_min")
    ap.add_argument("--snr-max", type=float, default=10.0, dest="snr_max")
    ap.add_argument("--qrm-prob", type=float, default=0.6, dest="qrm_prob")
    ap.add_argument("--flutter", action="store_true", help="enable channel flutter (extra realism)")
    ap.add_argument("--init", default="", help="warm-start weights from this checkpoint (fine-tune)")
    args = ap.parse_args()

    chan = ChannelConfig(flutter=True if args.flutter else ChannelConfig().flutter)
    gen = GenConfig(snr_range=(args.snr_min, args.snr_max), qrm_prob=args.qrm_prob, channel=chan)
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch, num_workers=args.workers,
                      time_downsample=args.downsample, width=args.width, out_dir=args.out, lr=args.lr,
                      wmr_dir=args.wmr, wmr_prob=args.wmr_prob, gen_config=gen,
                      init_ckpt=args.init)
    train(cfg)


if __name__ == "__main__":
    main()
