import torch
import torch.nn as nn
from deepfist.model.net import CwCtcNet, N_CLASSES, count_params


def test_forward_shape_downsample2():
    net = CwCtcNet(time_downsample=2)
    x = torch.randn(2, 1, 23, 751)
    lp = net(x)
    assert lp.shape[1] == 2 and lp.shape[2] == N_CLASSES
    assert 370 <= lp.shape[0] <= 380      # ~376
    # log-softmax: exp sums to ~1 across classes
    assert torch.allclose(lp.exp().sum(-1), torch.ones(lp.shape[0], 2), atol=1e-4)


def test_forward_shape_downsample4():
    net = CwCtcNet(time_downsample=4)
    lp = net(torch.randn(2, 1, 23, 751))
    assert 184 <= lp.shape[0] <= 192       # ~188


def test_gradient_flows_through_ctc():
    net = CwCtcNet()
    x = torch.randn(2, 1, 23, 400)
    lp = net(x)                            # [T,2,C]
    T = lp.shape[0]
    targets = torch.tensor([5, 6, 7, 8])   # two samples, len 2 each
    tgt_len = torch.tensor([2, 2])
    inp_len = torch.full((2,), T)
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(lp, targets, inp_len, tgt_len)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_param_count_is_small():
    n = count_params(CwCtcNet())
    assert 300_000 < n < 1_500_000
