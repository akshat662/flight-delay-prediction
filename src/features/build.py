"""Feature engineering, encoding, splits, and scaling.

Ports reference/FINAL.ipynb cells 31-69 (X/Y construction, label encoding,
75/25 split, Z-scaling, one-hot + MinMax alternate) and cells 83-87
(chronological LSTM sequence data). Input is
data/processed/flights_clean.parquet; every output frame is written to
data/processed/ and every fitted encoder/scaler to models/preprocessors/
via joblib.

Deliberate departures from the notebook (see module docstring notes inline
at each site):
  - CRS_ELAPSED_TIME, AIRLINE_DOT, AIRLINE_CODE, DOT_CODE, ORIGIN_CITY,
    DEST_CITY are excluded from X_df (redundancy findings from the feature-
    selection statistics in src.features.selection). The notebook already
    omitted these from its X_df selection, so this is a documentation
    change, not a behavior change.
  - The notebook's `df["DISTANCE"]` cross-frame cast (cell 33/40) relied on
    a pre-outlier-pruned `data` copy that no longer exists as a separate
    artifact in this pipeline (cleaning persists only the post-prune
    frame). Index-aligned assignment made the original behaviorally
    equivalent to casting X_df's own DISTANCE column, so that's what this
    module does.
  - Two commented-out, hardcoded-row-value asserts (shared-encoder spot
    check in cell 36; first-5-rows check in cell 38) are replaced with
    equivalent order-independent checks: a full round-trip check on the
    shared ORIGIN/DEST encoder, and a valid-minutes-range check on the
    converted times. Both are gated by --no-validate like the notebook's
    other TEST-flag checks.
  - Cell 69 saves the one-hot frames from *before* cell 67's MinMax step,
    silently discarding the normalized frames. This module saves the
    normalized frames instead (X_train_onehot_norm / X_test_onehot_norm),
    since that is what "one-hot + MinMax alternate" describes.
  - Outputs are saved as Parquet, per this project's standing convention,
    not the notebook's CSV.
  - Cell 58's `_cpy` aliases are unused within the ported cell range and
    are not ported.

Run as: python -m src.features.build [--force] [--no-validate]
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.config import Config, load_config
from src.utils.io import load_df, save_df
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

# Columns selected into X_df (cell 31). CRS_ELAPSED_TIME, AIRLINE_DOT,
# AIRLINE_CODE, DOT_CODE, ORIGIN_CITY, DEST_CITY are excluded per the
# feature-selection redundancy findings (src.features.selection).
X_SELECT_COLS = [
    "FL_DATE", "FL_NUMBER", "AIRLINE", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "TAXI_IN", "TAXI_OUT",
]

# Column order matches cell 31's Y_df exactly (not src.config's targets order).
Y_COLS = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_SECURITY", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]

NUMERIC_COLS = ["CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "TAXI_IN", "TAXI_OUT"]
CATEGORICAL_COLS = ["AIRLINE", "ORIGIN", "DEST", "YEAR", "MONTH", "DAY"]
ONE_HOT_COLS = ["YEAR", "MONTH", "DAY", "AIRLINE", "DEST", "ORIGIN"]
LSTM_FEATURE_COLS = [
    "YEAR", "MONTH", "DAY", "AIRLINE", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "TAXI_IN", "TAXI_OUT",
]
LSTM_SORT_COLS = ["YEAR", "MONTH", "DAY", "CRS_DEP_TIME", "CRS_ARR_TIME"]


def convert_hhmm_to_mins(hhmm_list) -> np.ndarray:
    """Convert hhmm-format time strings (no leading zeros) to minutes past midnight.

    "5" -> 0:05, "45" -> 0:45, "715" -> 7:15, "2359" -> 23:59.
    Raises ValueError on any string that isn't 1-4 characters long.
    """
    hours = []
    mins = []

    for s in hhmm_list:
        if len(s) == 1:
            hours.append(0)
            mins.append(int(s))
        elif len(s) == 2:
            hours.append(0)
            mins.append(int(s))
        elif len(s) == 3:
            hours.append(int(s[0]))
            mins.append(int(s[1:]))
        elif len(s) == 4:
            hours.append(int(s[:2]))
            mins.append(int(s[2:]))
        else:
            raise ValueError(s)

    return np.array(hours) * 60 + np.array(mins)


def build_xy(df: pd.DataFrame, arr_delay_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Select X_df, Y_df, arr_only_y_df from the cleaned frame (cell 31)."""
    X_df = df[X_SELECT_COLS].copy()
    Y_df = df[Y_COLS].copy()
    arr_only_y_df = df[arr_delay_col].copy()
    return X_df, Y_df, arr_only_y_df


def _validate_no_nulls(df: pd.DataFrame, name: str) -> None:
    """cells 34/48: assert a frame has no null values."""
    n_null = int(df.isna().sum().sum())
    assert n_null == 0, f"{name} contains {n_null} null values"
    logger.info("Validated: %s has no null values", name)


def encode_categoricals(
    X_df: pd.DataFrame, validate: bool = True
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """cell 36: label-encode AIRLINE; encode ORIGIN/DEST together with one shared encoder."""
    original_origin = X_df["ORIGIN"].copy()
    original_dest = X_df["DEST"].copy()

    airline_encoder = LabelEncoder()
    X_df["AIRLINE"] = airline_encoder.fit_transform(X_df["AIRLINE"])

    airport_encoder = LabelEncoder()
    airport_encoder.fit(pd.concat([X_df["ORIGIN"], X_df["DEST"]]))
    X_df["ORIGIN"] = airport_encoder.transform(X_df["ORIGIN"])
    X_df["DEST"] = airport_encoder.transform(X_df["DEST"])

    if validate:
        # Order-independent replacement for the notebook's hardcoded-row spot
        # check: the shared encoder must round-trip every ORIGIN/DEST value.
        assert (airport_encoder.inverse_transform(X_df["ORIGIN"]) == original_origin.to_numpy()).all()
        assert (airport_encoder.inverse_transform(X_df["DEST"]) == original_dest.to_numpy()).all()
        logger.info("Validated: shared ORIGIN/DEST encoder round-trips exactly")

    return X_df, airline_encoder, airport_encoder


def convert_times_to_minutes(X_df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """cell 38: convert CRS_DEP_TIME/CRS_ARR_TIME from hhmm to minutes past midnight."""
    X_df["CRS_DEP_TIME"] = X_df["CRS_DEP_TIME"].astype(str)
    X_df["CRS_DEP_TIME"] = convert_hhmm_to_mins(X_df["CRS_DEP_TIME"])

    X_df["CRS_ARR_TIME"] = X_df["CRS_ARR_TIME"].astype(str)
    X_df["CRS_ARR_TIME"] = convert_hhmm_to_mins(X_df["CRS_ARR_TIME"])

    if validate:
        # Order-independent replacement for the notebook's hardcoded
        # first-5-rows check: every converted time is a valid minute-of-day.
        # Upper bound is 1440, not 1439: BTS records a midnight-exactly
        # schedule as "2400" (a handful of rows), which converts to 1440.
        assert X_df["CRS_DEP_TIME"].between(0, 1440).all()
        assert X_df["CRS_ARR_TIME"].between(0, 1440).all()
        logger.info("Validated: converted CRS_DEP_TIME/CRS_ARR_TIME are within [0, 1440] minutes")

    return X_df


def split_date(X_df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """cell 42: break FL_DATE into YEAR/MONTH/DAY and drop FL_DATE."""
    X_df["FL_DATE"] = pd.to_datetime(X_df["FL_DATE"])
    X_df["YEAR"] = X_df["FL_DATE"].dt.year
    X_df["MONTH"] = X_df["FL_DATE"].dt.month
    X_df["DAY"] = X_df["FL_DATE"].dt.day
    X_df = X_df.drop(columns=["FL_DATE"])

    if validate:
        assert X_df["MONTH"].min() == 1
        assert X_df["MONTH"].max() == 12
        assert X_df["DAY"].min() == 1
        assert X_df["DAY"].max() == 31
        logger.info("Validated: MONTH in [1, 12], DAY in [1, 31]")

    return X_df


def _log_pca_check(X_train: pd.DataFrame, numeric_cols: list[str], Y_train: pd.DataFrame) -> None:
    """cells 61-63: PCA over the numeric columns, correlation vs summed Y_train. Informational only."""
    y_sum = Y_train.sum(axis=1).to_numpy()
    for n_components in range(1, len(numeric_cols)):
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_train[numeric_cols])
        corr_strs = []
        for i in range(n_components):
            r, _ = pearsonr(components[:, i], y_sum)
            corr_strs.append(f"PCA{i + 1}={r:.4f}")
        logger.info("PCA n_components=%d: %s", n_components, ", ".join(corr_strs))


def _log_lstm_group_sizes(X_train_lstm: pd.DataFrame) -> None:
    """cell 85: longest/shortest flights-per-day, to check whether day-padding makes sense. Informational only."""
    sizes = X_train_lstm.groupby(["YEAR", "MONTH", "DAY"]).size()
    logger.info("LSTM flights-per-day: longest=%d, shortest=%d", int(sizes.max()), int(sizes.min()))


def build(config: Config | None = None, force: bool = False, validate: bool = True) -> dict[str, pd.DataFrame]:
    """Run the full feature-engineering pipeline, writing every artifact under data/processed/ and models/preprocessors/."""
    config = config or load_config()

    if not force and config.paths.y_test_lstm.exists():
        logger.info("Feature-engineering outputs already present (force=False); skipping. Pass --force to rebuild.")
        return {
            "X_train_scale": load_df(config.paths.x_train_label),
            "X_test_scale": load_df(config.paths.x_test_label),
            "Y_train": load_df(config.paths.y_train),
            "Y_test": load_df(config.paths.y_test),
        }

    df = load_df(config.paths.clean_parquet)

    X_df, Y_df, arr_only_y_df = build_xy(df, config.arrival_delay_col)

    if validate:
        _validate_no_nulls(X_df, "X_df")

    X_df, airline_encoder, airport_encoder = encode_categoricals(X_df, validate=validate)
    X_df = convert_times_to_minutes(X_df, validate=validate)
    X_df["DISTANCE"] = X_df["DISTANCE"].astype(int)
    X_df = split_date(X_df, validate=validate)
    X_df = X_df.drop(columns=["FL_NUMBER"])
    X_df["TAXI_IN"] = X_df["TAXI_IN"].astype(int)
    X_df["TAXI_OUT"] = X_df["TAXI_OUT"].astype(int)
    logger.info("X_df built: %d rows x %d cols", *X_df.shape)

    if validate:
        _validate_no_nulls(Y_df, "Y_df")

    Y_df = Y_df.astype(int)
    arr_only_y_df = arr_only_y_df.astype(int)
    if validate:
        assert len(X_df) == len(Y_df)
        logger.info("Validated: len(X_df) == len(Y_df) == %d", len(X_df))

    # 75/25 split (cell 53)
    X_train, X_test, Y_train, Y_test, y_arr_only_train, y_arr_only_test = train_test_split(
        X_df, Y_df, arr_only_y_df, random_state=config.seed, test_size=config.split.test_size
    )
    logger.info("Split: X_train=%d rows, X_test=%d rows", len(X_train), len(X_test))

    # Z-scale the five numeric columns, fit on train only (cell 55)
    standard_scaler = StandardScaler()
    X_train_scale = pd.DataFrame(
        standard_scaler.fit_transform(X_train[NUMERIC_COLS]), columns=NUMERIC_COLS, index=X_train.index
    )
    X_test_scale = pd.DataFrame(
        standard_scaler.transform(X_test[NUMERIC_COLS]), columns=NUMERIC_COLS, index=X_test.index
    )
    X_train_scale = pd.concat([X_train[CATEGORICAL_COLS], X_train_scale], axis=1)
    X_test_scale = pd.concat([X_test[CATEGORICAL_COLS], X_test_scale], axis=1)

    save_df(X_train_scale, config.paths.x_train_label)
    save_df(X_test_scale, config.paths.x_test_label)
    save_df(Y_train, config.paths.y_train)
    save_df(Y_test, config.paths.y_test)
    save_df(y_arr_only_train.to_frame(config.arrival_delay_col), config.paths.arr_delay_y_train)
    save_df(y_arr_only_test.to_frame(config.arrival_delay_col), config.paths.arr_delay_y_test)

    _log_pca_check(X_train, NUMERIC_COLS, Y_train)

    # One-hot + MinMax alternate (cells 65, 67; re-split with the same seed/test_size)
    X_one_hot = pd.get_dummies(X_df, columns=ONE_HOT_COLS)
    X_train_oh, X_test_oh, _, _, _, _ = train_test_split(
        X_one_hot, Y_df, arr_only_y_df, random_state=config.seed, test_size=config.split.test_size
    )
    minmax_scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(
        minmax_scaler.fit_transform(X_train_oh), columns=X_train_oh.columns, index=X_train_oh.index
    )
    X_test_norm = pd.DataFrame(
        minmax_scaler.transform(X_test_oh), columns=X_test_oh.columns, index=X_test_oh.index
    )
    save_df(X_train_norm, config.paths.x_train_onehot_norm)
    save_df(X_test_norm, config.paths.x_test_onehot_norm)

    # Chronological (shuffle=False) sequence data for LSTM models (cells 83, 87)
    X_lstm1 = pd.concat([X_train_scale, Y_train], axis=1)
    X_lstm2 = pd.concat([X_test_scale, Y_test], axis=1)
    combined_df_lstm = pd.concat([X_lstm1, X_lstm2])
    combined_df_lstm = combined_df_lstm.sort_values(LSTM_SORT_COLS)

    X_LSTM = combined_df_lstm[LSTM_FEATURE_COLS]
    Y_LSTM = combined_df_lstm[Y_COLS]
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_LSTM, Y_LSTM, test_size=config.split.test_size, shuffle=False
    )
    _log_lstm_group_sizes(X_train_lstm)

    save_df(X_train_lstm, config.paths.x_train_lstm)
    save_df(X_test_lstm, config.paths.x_test_lstm)
    save_df(y_train_lstm, config.paths.y_train_lstm)
    save_df(y_test_lstm, config.paths.y_test_lstm)

    # Fitted encoders/scalers
    config.paths.preprocessors_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(airline_encoder, config.paths.airline_encoder)
    joblib.dump(airport_encoder, config.paths.airport_encoder)
    joblib.dump(standard_scaler, config.paths.standard_scaler)
    joblib.dump(minmax_scaler, config.paths.minmax_scaler)
    logger.info("Saved fitted encoders/scalers to %s", config.paths.preprocessors_dir)

    return {
        "X_train_scale": X_train_scale,
        "X_test_scale": X_test_scale,
        "Y_train": Y_train,
        "Y_test": Y_test,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering, encoding, splits, and scaling")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument("--no-validate", action="store_true", help="Skip the TEST-flag-equivalent assertions")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    set_all_seeds(42)
    build(force=args.force, validate=not args.no_validate)
