from src.matching.preprocess import clean


def test_lowercase():
    assert clean("Sony TURNTABLE") == "sony turntable"


def test_removes_punctuation_keeps_dash():
    assert clean("Sony PS-LX350H, (Belt)!") == "sony ps-lx350h belt"


def test_collapses_whitespace():
    assert clean("  sony    turntable  ") == "sony turntable"


def test_non_string_returns_empty():
    assert clean(None) == ""
    assert clean(float("nan")) == ""


def test_empty_string():
    assert clean("") == ""