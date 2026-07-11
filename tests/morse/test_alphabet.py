import pytest
from deepfist.morse import alphabet as ab


def test_basic_letter_pattern():
    assert ab.morse_for("A") == ".-"
    assert ab.morse_for("Q") == "--.-"


def test_prosigns_have_unique_patterns():
    assert ab.morse_for("<SK>") == "...-.-"
    assert ab.morse_for("<KN>") == "-.--."


def test_blank_and_space_are_classes():
    assert ab.TOKENS[0] == ab.BLANK
    assert ab.TOKENS[1] == ab.SPACE
    assert ab.morse_for(ab.SPACE) == ""


def test_text_to_tokens_roundtrip():
    toks = ab.text_to_tokens("CQ DE W1AW")
    assert toks == ["C", "Q", " ", "D", "E", " ", "W", "1", "A", "W"]
    assert ab.tokens_to_text(toks) == "CQ DE W1AW"


def test_prosign_and_alias_parsing():
    assert ab.text_to_tokens("R <SK>") == ["R", " ", "<SK>"]
    assert ab.text_to_tokens("<AR>") == ["+"]   # AR collapses to +
    assert ab.text_to_tokens("<BT>") == ["="]   # BT collapses to =


def test_unknown_symbol_raises():
    with pytest.raises(ValueError):
        ab.text_to_tokens("A#B")


def test_every_pattern_is_unique():
    patterns = list(ab.MORSE.values())
    assert len(patterns) == len(set(patterns))  # no two tokens share a code
