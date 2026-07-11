"""PARIS-standard CW element timing, with optional Farnsworth spacing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    dot: float          # seconds
    dash: float
    element_gap: float  # gap between elements within a character
    char_gap: float     # gap between characters
    word_gap: float     # gap between words


def wpm_to_timing(wpm: float, farnsworth_wpm: float | None = None) -> Timing:
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    dot = 1.2 / wpm
    char_gap = 3 * dot
    word_gap = 7 * dot
    if farnsworth_wpm is not None and 0 < farnsworth_wpm < wpm:
        # ARRL Farnsworth: total time uses effective wpm, elements use character wpm.
        t_char = 1.2 / wpm
        t_eff = 1.2 / farnsworth_wpm
        inter = (50 * t_eff - 31 * t_char) / 19  # seconds of "one gap unit" (3 dots nominal)
        char_gap = 3 * inter
        word_gap = 7 * inter
    return Timing(
        dot=dot, dash=3 * dot, element_gap=dot,
        char_gap=char_gap, word_gap=word_gap,
    )
