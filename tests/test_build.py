"""Unit tests for src.features.build.convert_hhmm_to_mins."""

from __future__ import annotations

import pytest

from src.features.build import convert_hhmm_to_mins


def test_convert_hhmm_to_mins_various_lengths():
    result = convert_hhmm_to_mins(["5", "45", "715", "2359"])
    assert list(result) == [5, 45, 435, 1439]


def test_convert_hhmm_to_mins_raises_on_five_chars():
    with pytest.raises(ValueError):
        convert_hhmm_to_mins(["12345"])
