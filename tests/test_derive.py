"""Boundary-value unit tests for src.features.derive."""

from __future__ import annotations

import pandas as pd

from src.features.derive import add_distance_bucket, add_season, add_time_of_day


def test_time_of_day_boundaries():
    df = pd.DataFrame({"CRS_DEP_TIME": [400, 1159, 1200, 2359], "CRS_ARR_TIME": [400, 1159, 1200, 2359]})
    out = add_time_of_day(df)
    assert list(out["dept_time_of_day"]) == ["Morning", "Morning", "Afternoon", "Night"]
    assert list(out["arr_time_of_day"]) == ["Morning", "Morning", "Afternoon", "Night"]


def test_season_boundaries():
    df = pd.DataFrame({"FL_DATE": pd.to_datetime(["2021-02-15", "2021-03-01", "2021-05-31", "2021-06-01"])})
    out = add_season(df)
    assert list(out["season"]) == ["Winter", "Spring", "Spring", "Summer"]


def test_distance_bucket_boundaries():
    df = pd.DataFrame({"DISTANCE": [0, 500, 501, 3500, 4983]})
    out = add_distance_bucket(df)
    assert list(out["distance_bucket"]) == [1, 1, 2, 4, 5]
