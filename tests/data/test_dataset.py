import itertools
import torch
from deepfist.data.dataset import CwIterableDataset, collate, make_loader


def test_yields_expected_shapes():
    ds = CwIterableDataset(base_seed=0)
    spec, target, tlen = next(iter(ds))
    assert spec.shape[0] == 1 and spec.shape[1] == 65
    assert target.dtype == torch.long and len(target) == tlen


def test_collate_flat_targets():
    ds = CwIterableDataset(base_seed=0)
    batch = list(itertools.islice(iter(ds), 4))
    specs, targets, tlens = collate(batch)
    assert specs.shape[0] == 4 and specs.shape[1] == 1
    assert int(tlens.sum()) == len(targets)


def test_seed_determinism_single_worker():
    a = list(itertools.islice(iter(CwIterableDataset(base_seed=7)), 3))
    b = list(itertools.islice(iter(CwIterableDataset(base_seed=7)), 3))
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))


def test_loader_produces_batch():
    loader = make_loader(base_seed=0, batch_size=4, num_workers=0)
    specs, targets, tlens = next(iter(loader))
    assert specs.shape[0] == 4
