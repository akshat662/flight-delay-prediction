"""Pure functions that attach the derived columns used by the source EDA notebook.

Each function takes a DataFrame and returns a new DataFrame with one added
column (or a small group of related columns). No other feature engineering
belongs here — later phases own encoding, scaling, and modeling features.
"""

from __future__ import annotations

import logging

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

TIME_OF_DAY_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

DISTANCE_BUCKET_EDGES_BASE = [0, 500, 1500, 2500, 3500]
DISTANCE_BUCKET_LABELS = [1, 2, 3, 4, 5]


def add_speed(df: pd.DataFrame) -> pd.DataFrame:
    """speed = DISTANCE / AIR_TIME."""
    df = df.copy()
    df["speed"] = df["DISTANCE"] / df["AIR_TIME"]
    return df


def add_year_month(df: pd.DataFrame, date_col: str = "FL_DATE") -> pd.DataFrame:
    """year = FL_DATE.dt.year, month = FL_DATE.dt.month."""
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    return df


def add_season(df: pd.DataFrame, date_col: str = "FL_DATE") -> pd.DataFrame:
    """season derived from FL_DATE's month."""
    df = df.copy()
    df["season"] = df[date_col].dt.month.map(_SEASON_BY_MONTH)
    return df


def add_weekday(df: pd.DataFrame, date_col: str = "FL_DATE") -> pd.DataFrame:
    """weekday = FL_DATE.dt.day_name()."""
    df = df.copy()
    df["weekday"] = df[date_col].dt.day_name()
    return df


def add_holiday_flag(df: pd.DataFrame, date_col: str = "FL_DATE") -> pd.DataFrame:
    """holiday = FL_DATE in holidays.US(), as a boolean."""
    df = df.copy()
    years = sorted(int(y) for y in df[date_col].dt.year.dropna().unique())
    us_holidays = holidays.US(years=years)
    df["holiday"] = df[date_col].dt.date.isin(us_holidays)
    return df


def _time_of_day(hhmm: pd.Series) -> pd.Series:
    hours = (hhmm // 100).astype(int)
    conditions = [
        (hours >= 4) & (hours < 12),
        (hours >= 12) & (hours < 16),
        (hours >= 16) & (hours < 20),
    ]
    choices = ["Morning", "Afternoon", "Evening"]
    return pd.Series(np.select(conditions, choices, default="Night"), index=hhmm.index)


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """dept_time_of_day / arr_time_of_day from CRS_DEP_TIME / CRS_ARR_TIME (hhmm ints)."""
    df = df.copy()
    df["dept_time_of_day"] = _time_of_day(df["CRS_DEP_TIME"])
    df["arr_time_of_day"] = _time_of_day(df["CRS_ARR_TIME"])
    return df


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """distance_bucket: DISTANCE binned on [0, 500, 1500, 2500, 3500, max], labelled 1-5."""
    df = df.copy()
    edges = DISTANCE_BUCKET_EDGES_BASE + [df["DISTANCE"].max()]
    df["distance_bucket"] = pd.cut(
        df["DISTANCE"], bins=edges, labels=DISTANCE_BUCKET_LABELS, include_lowest=True
    )
    return df


def add_taxi_flags(df: pd.DataFrame) -> pd.DataFrame:
    """taxi_in_flag / taxi_out_flag: 'High' if TAXI_IN > 20 / TAXI_OUT > 30 else 'Low'."""
    df = df.copy()
    df["taxi_in_flag"] = np.where(df["TAXI_IN"] > 20, "High", "Low")
    df["taxi_out_flag"] = np.where(df["TAXI_OUT"] > 30, "High", "Low")
    return df


def add_all_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every derived column in one pass, for callers that need the full set."""
    df = add_speed(df)
    df = add_year_month(df)
    df = add_season(df)
    df = add_weekday(df)
    df = add_holiday_flag(df)
    df = add_time_of_day(df)
    df = add_distance_bucket(df)
    df = add_taxi_flags(df)
    logger.info("Attached derived columns to %d rows", len(df))
    return df
