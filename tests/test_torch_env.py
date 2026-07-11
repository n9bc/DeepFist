import torch


def test_torch_imports_and_computes():
    x = torch.ones(3)
    assert torch.equal(x + x, torch.full((3,), 2.0))
    assert torch.__version__.split(".")[0] == "2"


def test_cuda_report():
    # Does not require CUDA (CI may be CPU) — just must not error.
    _ = torch.cuda.is_available()
