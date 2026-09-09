"""Clean the raw flight-delay CSV into the modeling-ready Parquet artifact.

Pipeline (run in exactly this order, row counts logged at every step):
    1. Read CSV, parse FL_DATE.
    2. Record cancelled/diverted monthly counts, then drop those rows.
    3. Drop duplicates.
    4. Split rows by presence of the five DELAY_DUE_* component columns.
    5. Keep only rows with complete delay components.
    6. Integrity check: components must sum exactly to ARR_DELAY.
    7. IQR outlier pruning on ARR_DELAY.
    8. Save to data/processed/flights_clean.parquet.

Run as: python -m src.data.clean [--force]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import Config, load_config
from src.utils.io import cached, load_df, save_df

logger = logging.getLogger(__name__)


def _log_step(step: str, df: pd.DataFrame) -> None:
    logger.info("[%s] row count: %d", step, len(df))


def load_nocancel(
    config: Config | None = None, force: bool = False, stats: dict | None = None
) -> pd.DataFrame:
    """Return the post-cancellation-drop, pre-component-filter frame (steps 1-2).

    Reads the raw CSV, records monthly cancelled/diverted counts to
    data/interim/cancelled_diverted_monthly.parquet, drops cancelled/diverted
    rows, and caches the result to data/interim/flights_nocancel.parquet.
    `clean()` and Phase 2 EDA both build on this frame so the read+drop logic
    lives in exactly one place.
    """
    config = config or load_config()
    out_path = config.paths.flights_nocancel

    if cached(out_path, force=force):
        df = load_df(out_path)
        if stats is not None:
            stats["n_read"] = None
            stats["n_after_drop"] = len(df)
        return df

    logger.info("Reading CSV from %s", config.paths.raw_csv)
    df = pd.read_csv(config.paths.raw_csv, parse_dates=["FL_DATE"], date_format="%Y-%m-%d")
    _log_step("1_read_csv", df)
    n_read = len(df)

    cancelled_or_diverted = (df["CANCELLED"] == 1) | (df["DIVERTED"] == 1)
    monthly = (
        df.loc[cancelled_or_diverted]
        .assign(year_month=df.loc[cancelled_or_diverted, "FL_DATE"].dt.to_period("M").astype(str))
        .groupby("year_month")
        .size()
        .rename("cancelled_diverted_count")
    )
    save_df(monthly.reset_index(), config.paths.cancelled_diverted_monthly)

    df = df.loc[~cancelled_or_diverted].copy()
    _log_step("2_drop_cancelled_diverted", df)

    save_df(df, out_path)

    if stats is not None:
        stats["n_read"] = n_read
        stats["n_after_drop"] = len(df)

    return df


def split_by_components(
    df: pd.DataFrame, target_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by whether the five DELAY_DUE_* columns are present.

    Verifies and logs that these columns are all-present or all-missing per
    row (never partial) before splitting. Returns (has_components, missing_components).
    """
    na_mask = df[target_cols].isna()
    n_na_per_row = na_mask.sum(axis=1)
    partial_mask = (n_na_per_row > 0) & (n_na_per_row < len(target_cols))
    n_partial = int(partial_mask.sum())
    if n_partial > 0:
        logger.warning(
            "%d rows have a PARTIAL set of delay-component values (expected all-or-nothing)",
            n_partial,
        )
    else:
        logger.info("Verified: delay-component columns are all-present or all-missing per row")

    complete_mask = n_na_per_row == 0
    has_components = df[complete_mask]
    missing_components = df[~complete_mask]
    return has_components, missing_components


def _check_components_sum_to_arr_delay(
    df: pd.DataFrame, target_cols: list[str], arr_delay_col: str
) -> pd.DataFrame:
    """Vectorized integrity check: components must sum exactly to ARR_DELAY."""
    component_sum = df[target_cols].sum(axis=1)
    matches = component_sum == df[arr_delay_col]
    n_violations = int((~matches).sum())
    logger.info("Component-sum integrity check: %d violations out of %d rows", n_violations, len(df))
    assert n_violations == 0, f"{n_violations} rows where delay components do not sum to {arr_delay_col}"
    return df


def _iqr_prune(
    df: pd.DataFrame, col: str, multiplier: float
) -> tuple[pd.DataFrame, dict]:
    """Prune rows outside [Q1 - m*IQR, Q3 + m*IQR] on `col`. Returns (pruned_df, stats)."""
    before_desc = df[col].describe()

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (df[col] >= lower) & (df[col] <= upper)
    pruned = df[mask]
    after_desc = pruned[col].describe()

    n_removed = len(df) - len(pruned)
    frac_removed = n_removed / len(df) if len(df) else 0.0

    logger.info("IQR bounds on %s: Q1=%.2f Q3=%.2f IQR=%.2f lower=%.2f upper=%.2f", col, q1, q3, iqr, lower, upper)
    logger.info("Before pruning:\n%s", before_desc)
    logger.info("After pruning:\n%s", after_desc)
    logger.info("Removed %d rows (%.4f%% of input)", n_removed, frac_removed * 100)

    stats = {
        "before": before_desc,
        "after": after_desc,
        "n_removed": n_removed,
        "frac_removed": frac_removed,
    }
    return pruned, stats


def clean(config: Config | None = None, force: bool = False) -> pd.DataFrame:
    """Run the full cleaning pipeline, caching to data/processed/flights_clean.parquet."""
    config = config or load_config()
    out_path = config.paths.clean_parquet

    if cached(out_path, force=force):
        return load_df(out_path)

    target_cols = config.targets
    arr_delay_col = config.arrival_delay_col
    summary_lines: list[str] = ["# Cleaning Summary", ""]

    # Steps 1-2: read CSV and drop cancelled/diverted rows (shared with EDA's FRAME A)
    load_stats: dict = {}
    df = load_nocancel(config, force=force, stats=load_stats)
    if load_stats.get("n_read") is not None:
        summary_lines.append(f"1. Read CSV: {load_stats['n_read']} rows")
    else:
        summary_lines.append("1. Read CSV: (reused cached nocancel frame)")
    summary_lines.append(f"2. Drop cancelled/diverted: {load_stats['n_after_drop']} rows")

    # Step 3: drop duplicates
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before_dedup - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d duplicate rows (expected 0)", n_dropped)
    _log_step("3_drop_duplicates", df)
    summary_lines.append(f"3. Drop duplicates: {len(df)} rows ({n_dropped} duplicates removed)")

    # Step 4: split by presence of delay components
    has_components, missing_components = split_by_components(df, target_cols)
    logger.info(
        "has_components: %d, missing_components: %d", len(has_components), len(missing_components)
    )
    summary_lines.append(
        f"4. Split by delay components: has_components={len(has_components)}, "
        f"missing_components={len(missing_components)}"
    )

    # Step 5: keep only rows with complete delay components
    df = has_components
    retained_fraction = len(df) / n_before_dedup if n_before_dedup else 0.0
    _log_step("5_keep_has_components", df)
    logger.info("Retained fraction of non-cancelled rows: %.4f", retained_fraction)
    summary_lines.append(f"5. Keep has_components: {len(df)} rows (retained fraction {retained_fraction:.4f})")

    # Step 6: integrity check — components sum to ARR_DELAY
    df = _check_components_sum_to_arr_delay(df, target_cols, arr_delay_col)
    _log_step("6_integrity_check", df)
    summary_lines.append(f"6. Integrity check (components sum to {arr_delay_col}): 0 violations, {len(df)} rows")

    # Step 7: IQR outlier pruning on ARR_DELAY
    df, iqr_stats = _iqr_prune(df, arr_delay_col, config.cleaning.iqr_multiplier)
    _log_step("7_iqr_prune", df)
    summary_lines.append(f"7. IQR prune on {arr_delay_col}: {len(df)} rows")
    summary_lines.append("")
    summary_lines.append(f"## {arr_delay_col} before/after IQR pruning")
    summary_lines.append("")
    summary_lines.append("| stat | before | after |")
    summary_lines.append("|---|---|---|")
    for stat in ["count", "mean", "std", "min", "max"]:
        summary_lines.append(
            f"| {stat} | {iqr_stats['before'][stat]:.2f} | {iqr_stats['after'][stat]:.2f} |"
        )
    summary_lines.append("")
    summary_lines.append(
        f"Removed {iqr_stats['n_removed']} rows ({iqr_stats['frac_removed'] * 100:.2f}%)"
    )

    # Step 8: save
    save_df(df, out_path)
    summary_lines.append("")
    summary_lines.append(f"8. Saved {len(df)} rows to {out_path}")

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    config.paths.cleaning_summary_md.write_text("\n".join(summary_lines) + "\n")
    logger.info("Wrote cleaning summary to %s", config.paths.cleaning_summary_md)

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the flight-delay dataset")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    clean(force=args.force)
