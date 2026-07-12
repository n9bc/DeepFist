"""Infinite on-the-fly labeled CW dataset wrapping the synthetic generator."""
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from deepfist.synth.generator import generate, GenConfig
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID


class CwIterableDataset(IterableDataset):
    def __init__(self, base_seed: int = 0, gen_config: GenConfig | None = None):
        self.base_seed = base_seed
        self.gen_config = gen_config or GenConfig()

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        step = 0
        sr = self.gen_config.sample_rate
        while True:
            seed = self.base_seed * 1_000_003 + wid + step * nworkers
            s = generate(seed=seed, config=self.gen_config)
            ids = [TOKEN_TO_ID[t] for t in text_to_tokens(s.label)]
            spec = audio_to_spectrogram(s.audio, sr).unsqueeze(0)   # [1,F,T]
            if spec.shape[-1] >= max(1, len(ids)):                  # CTC length guard
                yield spec, torch.tensor(ids, dtype=torch.long), len(ids)
            step += 1


def collate(batch):
    specs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return specs, targets, target_lengths


def _worker_init(_worker_id):
    # Pin each data worker to a single thread: numpy/torch BLAS otherwise spawn
    # many threads per worker and oversubscribe the CPU, starving the GPU.
    torch.set_num_threads(1)


def make_loader(base_seed: int = 0, batch_size: int = 128,
                num_workers: int = 0, gen_config: GenConfig | None = None) -> DataLoader:
    ds = CwIterableDataset(base_seed=base_seed, gen_config=gen_config)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=collate, drop_last=True,
                      worker_init_fn=_worker_init if num_workers > 0 else None,
                      persistent_workers=num_workers > 0)
