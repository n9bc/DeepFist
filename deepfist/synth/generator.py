"""Orchestrate morse + synth units into fixed-length labeled CW clips."""
from dataclasses import dataclass, field
import numpy as np

from deepfist.synth.text import random_message
from deepfist.synth.keyer import text_to_segments, segments_to_envelope
from deepfist.synth.fist import apply_fist, FistParams
from deepfist.synth.tone import envelope_to_audio
from deepfist.synth.channel import degrade, ChannelConfig
from deepfist.morse.alphabet import text_to_tokens, tokens_to_text
from deepfist.morse.timing import wpm_to_timing


@dataclass
class GenConfig:
    sample_rate: int = 3200        # DeepCW-style hi-res front-end (12.5 Hz bins)
    window_s: float = 6.0
    wpm_range: tuple[float, float] = (10.0, 55.0)  # 40->55: cover fast contest CW (real clips ran >40)
    pitch_range: tuple[float, float] = (500.0, 760.0)   # matches measured on-air spread
    snr_range: tuple[float, float] = (-6.0, 10.0)
    impair: bool = True
    qrm: bool = True                # mix in weaker interfering CW signals
    qrm_prob: float = 0.6
    qrm_max: int = 3
    channel: ChannelConfig = field(default_factory=ChannelConfig)


@dataclass
class Sample:
    audio: np.ndarray
    label: str
    meta: dict


def _fit_message(rng, timing, window_s):
    """Draw a message and drop trailing tokens until its keyed duration fits.

    A single character always fits the window at 10-40 WPM, so this terminates.
    """
    msg = random_message(rng, max_tokens=40)
    toks = text_to_tokens(msg)
    while toks:
        candidate = tokens_to_text(toks).strip()
        if not candidate:
            break
        segs = text_to_segments(candidate, timing)
        dur = sum(s.duration for s in segs)
        if dur <= window_s * 0.9:
            return candidate, segs, dur
        toks = toks[:-1]
    # Fallback: shortest possible message.
    segs = text_to_segments("E", timing)
    return "E", segs, sum(s.duration for s in segs)


def _trim_segments(segs, max_s):
    """Keep only enough leading segments to fill max_s seconds (avoids rendering
    a 30 s interferer message just to truncate it to the 6 s window)."""
    out, acc = [], 0.0
    for s in segs:
        out.append(s)
        acc += s.duration
        if acc >= max_s:
            break
    return out


def _render_cw(rng, text, wpm, pitch, sr, n, drift_max):
    """Render text as keyed CW audio placed at a random offset in an n-sample clip."""
    timing = wpm_to_timing(wpm)
    segs = _trim_segments(text_to_segments(text, timing), n / sr)
    segs = apply_fist(segs, rng, FistParams(
        jitter_sigma=float(rng.uniform(0.03, 0.15)),
        weight=float(rng.uniform(-0.15, 0.15))))
    env = segments_to_envelope(segs, sr)
    drift = float(rng.uniform(0, drift_max)) if drift_max else 0.0
    audio = envelope_to_audio(env, sr, pitch, drift_hz=drift)
    if len(audio) > n:
        audio = audio[:n]
    clip = np.zeros(n, dtype=np.float32)
    start = int(rng.integers(0, n - len(audio) + 1)) if len(audio) < n else 0
    clip[start:start + len(audio)] = audio
    return clip


def _add_qrm(clip, rng, cfg, main_pitch, n):
    """Mix in 1..qrm_max weaker interfering CW signals at nearby pitches.

    Interferers are strictly quieter than the main signal, so the learnable
    rule is "transcribe the strongest CW, ignore the rest" — matching how you
    tune the target station to the center of a pileup.
    """
    k = int(rng.integers(1, cfg.qrm_max + 1))
    sr = cfg.sample_rate
    for _ in range(k):
        wpm = float(rng.uniform(*cfg.wpm_range))
        offset = float(rng.uniform(60, 350)) * (1 if rng.random() < 0.5 else -1)
        pitch = float(np.clip(main_pitch + offset, 350, 950))
        text = random_message(rng, max_tokens=40)
        inter = _render_cw(rng, text, wpm, pitch, sr, n,
                           drift_max=3.0 if cfg.impair else 0.0)
        clip = clip + float(rng.uniform(0.2, 0.65)) * inter
    return clip, k


def generate(seed: int | None = None, config: GenConfig | None = None) -> Sample:
    cfg = config or GenConfig()
    rng = np.random.default_rng(seed)
    sr = cfg.sample_rate
    n = int(cfg.window_s * sr)

    wpm = float(rng.uniform(*cfg.wpm_range))
    pitch = float(rng.uniform(*cfg.pitch_range))
    snr = float(rng.uniform(*cfg.snr_range))
    timing = wpm_to_timing(wpm)

    msg, segs, keyed_dur = _fit_message(rng, timing, cfg.window_s)
    segs = apply_fist(segs, rng, FistParams(
        jitter_sigma=float(rng.uniform(0.03, 0.15)),
        weight=float(rng.uniform(-0.15, 0.15)),
    ))
    env = segments_to_envelope(segs, sr)
    drift = float(rng.uniform(0, 3)) if cfg.impair else 0.0
    audio = envelope_to_audio(env, sr, pitch, drift_hz=drift)
    if len(audio) > n:
        audio = audio[:n]

    # Random start offset within the window, then pad to fixed length.
    clip = np.zeros(n, dtype=np.float32)
    start = int(rng.integers(0, n - len(audio) + 1)) if len(audio) < n else 0
    clip[start:start + len(audio)] = audio

    n_qrm = 0
    if cfg.impair and cfg.qrm and rng.random() < cfg.qrm_prob:
        clip, n_qrm = _add_qrm(clip, rng, cfg, pitch, n)

    if cfg.impair:
        clip = degrade(clip, sr, snr, rng, cfg.channel, pitch_hz=pitch)

    meta = {
        "wpm": wpm, "pitch_hz": pitch, "snr_db": snr, "sample_rate": sr,
        "keyed_duration_s": keyed_dur, "window_s": cfg.window_s, "seed": seed,
        "start_s": start / sr, "n_qrm": n_qrm,
    }
    return Sample(audio=clip, label=msg, meta=meta)
