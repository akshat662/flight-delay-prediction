"""Unit tests for src.data.clean on small synthetic frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import split_by_components

TARGET_COLS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]


def test_split_by_components_all_or_nothing():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "DELAY_DUE_CARRIER": [5, np.nan, 0],
            "DELAY_DUE_WEATHER": [0, np.nan, 0],
            "DELAY_DUE_NAS": [0, np.nan, 0],
            "DELAY_DUE_SECURITY": [0, np.nan, 0],
            "DELAY_DUE_LATE_AIRCRAFT": [0, np.nan, 10],
        }
    )
    has_components, missing_components = split_by_components(df, TARGET_COLS)
    assert list(has_components["id"]) == [1, 3]
    assert list(missing_components["id"]) == [2]


def test_split_by_components_flags_partial_rows(caplog):
    df = pd.DataFrame(
        {
            "id": [1],
            "DELAY_DUE_CARRIER": [5],
            "DELAY_DUE_WEATHER": [np.nan],
            "DELAY_DUE_NAS": [0],
            "DELAY_DUE_SECURITY": [0],
            "DELAY_DUE_LATE_AIRCRAFT": [0],
        }
    )
    with caplog.at_level("WARNING"):
        has_components, missing_components = split_by_components(df, TARGET_COLS)
    assert "PARTIAL" in caplog.text
    # partial row (1 of 5 NaN) counts as "missing" (not complete)
    assert len(has_components) == 0
    assert len(missing_components) == 1


def test_components_sum_to_arr_delay_passes():
    df = pd.DataFrame(
        {
            "ARR_DELAY": [15, 0],
            "DELAY_DUE_CARRIER": [5, 0],
            "DELAY_DUE_WEATHER": [0, 0],
            "DELAY_DUE_NAS": [10, 0],
            "DELAY_DUE_SECURITY": [0, 0],
            "DELAY_DUE_LATE_AIRCRAFT": [0, 0],
        }
    )
    component_sum = df[TARGET_COLS].sum(axis=1)
    assert (component_sum == df["ARR_DELAY"]).all()


def test_components_sum_to_arr_delay_detects_violation():
    df = pd.DataFrame(
        {
            "ARR_DELAY": [15, 99],
            "DELAY_DUE_CARRIER": [5, 0],
            "DELAY_DUE_WEATHER": [0, 0],
            "DELAY_DUE_NAS": [10, 0],
            "DELAY_DUE_SECURITY": [0, 0],
            "DELAY_DUE_LATE_AIRCRAFT": [0, 0],
        }
    )
    component_sum = df[TARGET_COLS].sum(axis=1)
    matches = component_sum == df["ARR_DELAY"]
    assert not matches.all()
    assert (~matches).sum() == 1


def test_iqr_bounds_on_known_array():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # known quartiles for this series (linear interpolation, pandas default)
    assert q1 == pytest.approx(3.25)
    assert q3 == pytest.approx(7.75)
    assert iqr == pytest.approx(4.5)

    mask = (values >= lower) & (values <= upper)
    # 100 is a clear outlier and should be excluded
    assert 100 not in values[mask].values
    assert mask.sum() == 9


def test_cancelled_diverted_rows_are_dropped():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "CANCELLED": [0, 1, 0, 0],
            "DIVERTED": [0, 0, 1, 0],
        }
    )
    cancelled_or_diverted = (df["CANCELLED"] == 1) | (df["DIVERTED"] == 1)
    remaining = df.loc[~cancelled_or_diverted]
    assert list(remaining["id"]) == [1, 4]
