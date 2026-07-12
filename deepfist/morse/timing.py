"""PARIS-standard CW element timing, with optional Farnsworth spacing."""
from dataclasses import dataclass, replace


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


def morph_timing(timing: Timing, dahdit_ratio: float | None = None,
                 gap_scale: float = 1.0) -> Timing:
    """Return a copy of `timing` with a non-nominal dah/dit ratio and/or scaled gaps.

    Real fists and rigs don't hold the textbook 1:3 dot:dash ratio or exact PARIS
    spacing: measured dah/dit ratios run ~2.6-3.4 and inter-element/character gaps
    stretch or compress (~0.65-1.15x). Varying these at *generation* time (as
    opposed to the per-element noise `apply_fist` adds) reshapes the CW morphology
    the model must recognise. Defaults (ratio=None, gap_scale=1.0) leave `timing`
    unchanged, so callers that don't opt in keep the current behaviour.
    """
    dash = timing.dash if dahdit_ratio is None else dahdit_ratio * timing.dot
    if gap_scale == 1.0 and dahdit_ratio is None:
        return timing
    return replace(
        timing,
        dash=dash,
        element_gap=timing.element_gap * gap_scale,
        char_gap=timing.char_gap * gap_scale,
        word_gap=timing.word_gap * gap_scale,
    )
