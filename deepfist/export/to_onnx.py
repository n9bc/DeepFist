"""Export the CNN+CTC model to ONNX (spectrogram-in, dynamic batch+time)."""
import torch


def export_onnx(model, out_path: str, example_time: int = 751, opset: int = 17) -> None:
    assert not model.training, "call model.eval() before export (folds BatchNorm)"
    example = torch.randn(1, 1, 23, example_time)
    torch.onnx.export(
        model, example, out_path,
        input_names=["spectrogram"], output_names=["log_probs"],
        dynamic_axes={"spectrogram": {0: "batch", 3: "time"},
                      "log_probs": {0: "time_out", 1: "batch"}},
        opset_version=opset,
    )
