"""Blend WebMorseRunner labeled clips (from disk) with the on-the-fly numpy
generator. WMR gives real callsigns + validated keying; numpy gives infinite
variety. Both resolve to (spectrogram, token-ids, length) at 3200 Hz."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from deepfist.synth.generator import generate, GenConfig
from deepfist.features.spectrogram import audio_to_spectrogram, SAMPLE_RATE
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID
from deepfist.data.dataset import collate, _worker_init


def load_wmr_index(wmr_dir: str):
    """Return [(wav_path, token_ids)] for clips whose label tokenizes cleanly."""
    d = Path(wmr_dir)
    items = []
    for line in (d / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        try:
            ids = [TOKEN_TO_ID[t] for t in text_to_tokens(r["text"])]
        except KeyError:
            continue
        if ids:
            items.append((str(d / r["file"]), ids))
    return items


def _wmr_spec(wav_path):
    sr, a = wavfile.read(wav_path)
    a = a.astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return audio_to_spectrogram(a, SAMPLE_RATE).unsqueeze(0)   # [1,F,T]


class BlendedDataset(IterableDataset):
    def __init__(self, wmr_dir: str, wmr_prob: float = 0.5,
                 base_seed: int = 0, gen_config: GenConfig | None = None):
        self.items = load_wmr_index(wmr_dir)
        self.wmr_prob = wmr_prob
        self.base_seed = base_seed
        self.cfg = gen_config or GenConfig()
        if not self.items:
            raise RuntimeError(f"no usable WMR clips in {wmr_dir}")

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        rng = np.random.default_rng(self.base_seed * 7919 + wid)
        step = 0
        while True:
            if rng.random() < self.wmr_prob:
                path, ids = self.items[rng.integers(len(self.items))]
                spec = _wmr_spec(path)
            else:
                seed = self.base_seed * 1_000_003 + wid + step * nworkers
                s = generate(seed=seed, config=self.cfg)
                ids = [TOKEN_TO_ID[t] for t in text_to_tokens(s.label)]
                spec = audio_to_spectrogram(s.audio, self.cfg.sample_rate).unsqueeze(0)
                step += 1
            if spec.shape[-1] >= max(1, len(ids)):
                yield spec, torch.tensor(ids, dtype=torch.long), len(ids)


def make_blended_loader(wmr_dir: str, wmr_prob: float = 0.5, base_seed: int = 1,
                        batch_size: int = 128, num_workers: int = 0,
                        gen_config: GenConfig | None = None) -> DataLoader:
    ds = BlendedDataset(wmr_dir, wmr_prob, base_seed, gen_config)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=collate, drop_last=True,
                      worker_init_fn=_worker_init if num_workers > 0 else None,
                      persistent_workers=num_workers > 0)
