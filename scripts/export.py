"""Export a trained checkpoint to ONNX + metadata sidecar."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfist.export.to_onnx import export_from_checkpoint


def run_export(ckpt: str, out: str, downsample: int = 2) -> None:
    export_from_checkpoint(ckpt, out, time_downsample=downsample)
    print(f"exported: {out}")
    print(f"metadata: {out}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/exp1/model.pt")
    ap.add_argument("--out", default="deepfist.onnx")
    ap.add_argument("--downsample", type=int, default=2)
    args = ap.parse_args()
    run_export(args.ckpt, args.out, args.downsample)


if __name__ == "__main__":
    main()
