"""Training loop and the overfit-tiny wiring gate."""
import math
import os
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from deepfist.synth.generator import generate, GenConfig
from deepfist.features.spectrogram import audio_to_spectrogram
from deepfist.morse.alphabet import text_to_tokens, TOKEN_TO_ID
from deepfist.model.net import CwCtcNet
from deepfist.model.decode import greedy_ctc_decode
from deepfist.data.dataset import make_loader
from deepfist.train.metrics import cer, evaluate_per_snr


@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 128
    num_workers: int = 8
    lr: float = 3e-4
    warmup: int = 1000
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    eval_every: int = 2000
    time_downsample: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/exp"
    gen_config: GenConfig = field(default_factory=GenConfig)


def _encode(clip):
    ids = [TOKEN_TO_ID[t] for t in text_to_tokens(clip.label)]
    return torch.tensor(ids, dtype=torch.long)


def overfit_tiny(n_clips: int = 6, max_steps: int = 1000, device=None,
                 time_downsample: int = 2):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GenConfig(impair=False)
    clips = [generate(seed=s, config=cfg) for s in range(n_clips)]
    specs = torch.stack([audio_to_spectrogram(c.audio).unsqueeze(0) for c in clips]).to(device)
    targets = [_encode(c) for c in clips]
    target_lengths = torch.tensor([len(t) for t in targets])
    targets_cat = torch.cat(targets).to(device)

    net = CwCtcNet(time_downsample=time_downsample).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    net.train()
    last = 1e9
    for step in range(max_steps):
        opt.zero_grad()
        lp = net(specs)
        inp_len = torch.full((n_clips,), lp.shape[0], dtype=torch.long)
        loss = ctc(lp, targets_cat, inp_len, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        last = float(loss.item())
        if last < 0.05:
            net.eval()
            with torch.no_grad():
                preds = greedy_ctc_decode(net(specs))
            net.train()
            mean_cer = sum(cer(p, c.label) for p, c in zip(preds, clips)) / n_clips
            if mean_cer == 0.0:
                return last, 0.0, step
    net.eval()
    with torch.no_grad():
        preds = greedy_ctc_decode(net(specs))
    mean_cer = sum(cer(p, c.label) for p, c in zip(preds, clips)) / n_clips
    return last, mean_cer, max_steps


def _lr_at(step, warmup, total, base_lr, floor_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor_lr + 0.5 * (base_lr - floor_lr) * (1 + math.cos(math.pi * min(1.0, p)))


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = cfg.device
    net = CwCtcNet(time_downsample=cfg.time_downsample).to(device)
    decay, no_decay = [], []
    for _name, p in net.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.999))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    loader = make_loader(base_seed=1, batch_size=cfg.batch_size,
                         num_workers=cfg.num_workers, gen_config=cfg.gen_config)
    use_amp = device == "cuda"
    net.train()
    it = iter(loader)
    for step in range(cfg.steps):
        specs, targets, tlens = next(it)
        specs, targets, tlens = specs.to(device), targets.to(device), tlens.to(device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg.warmup, cfg.steps, cfg.lr, 1e-5)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lp = net(specs)
        inp_len = torch.full((specs.shape[0],), lp.shape[0], dtype=torch.long, device=device)
        loss = ctc(lp, targets, inp_len, tlens)      # lp already fp32 (log_softmax .float())
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {float(loss):.3f} lr {opt.param_groups[0]['lr']:.2e}")
        if step > 0 and step % cfg.eval_every == 0:
            table = evaluate_per_snr(net, [10, 6, 3, 0, -3, -6], 20,
                                     cfg.gen_config, device=device)
            print(f"[eval @ {step}] per-SNR CER: " +
                  " ".join(f"{k:+.0f}dB={v:.3f}" for k, v in table.items()))
            torch.save(net.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
            net.train()
    torch.save(net.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
