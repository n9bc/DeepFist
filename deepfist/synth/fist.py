"""Human keying imperfection: correlated timing jitter and weight drift."""
from dataclasses import dataclass
import numpy as np

from deepfist.synth.keyer import Segment


@dataclass
class FistParams:
    jitter_sigma: float = 0.08   # per-element multiplicative std-dev
    weight: float = 0.0          # -0.3..0.3; + lengthens ON at OFF's expense


def apply_fist(segments: list[Segment], rng: np.random.Generator,
               params: FistParams) -> list[Segment]:
    # One correlated per-clip bias (this operator's "fist"), plus per-element noise.
    clip_bias = rng.normal(0.0, params.jitter_sigma * 0.5) if params.jitter_sigma else 0.0
    out: list[Segment] = []
    for s in segments:
        factor = 1.0 + clip_bias
        if params.jitter_sigma:
            factor += rng.normal(0.0, params.jitter_sigma)
        if params.weight:
            factor += params.weight if s.on else -params.weight
        dur = max(1e-4, s.duration * factor)
        out.append(Segment(s.on, dur))
    return out
