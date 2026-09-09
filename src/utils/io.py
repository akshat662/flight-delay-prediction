"""Parquet I/O helpers and a simple recompute-skip cache guard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def save_df(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to Parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows x %d cols to %s", len(df), df.shape[1], path)


def load_df(path: Path | str) -> pd.DataFrame:
    """Load a DataFrame from Parquet."""
    path = Path(path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], path)
    return df


def cached(path: Path | str, force: bool = False) -> bool:
    """Return True if a cached artifact at `path` exists and should be reused.

    Usage:
        if cached(out_path, force=args.force):
            return load_df(out_path)
        df = expensive_compute(...)
        save_df(df, out_path)
    """
    path = Path(path)
    exists = path.exists()
    if exists and not force:
        logger.info("Using cached artifact at %s (force=False)", path)
        return True
    if exists and force:
        logger.info("Cached artifact at %s exists but force=True; recomputing", path)
    return False


def run_cached(
    path: Path | str,
    compute_fn: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """Load `path` if cached (and not force), else compute, save, and return it."""
    path = Path(path)
    if cached(path, force=force):
        return load_df(path)
    df = compute_fn()
    save_df(df, path)
    return df
