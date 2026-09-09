"""Typed, frozen configuration loaded from configs/config.yaml.

All paths in the resulting Config object are absolute, resolved against the
repository root, so no other module needs to reason about the working
directory it was launched from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    kaggle_slug: str
    filename: str


@dataclass(frozen=True)
class PathsConfig:
    data_raw_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    models_dir: Path
    preprocessors_dir: Path
    reports_dir: Path
    figures_dir: Path
    raw_csv: Path
    cancelled_diverted_monthly: Path
    flights_nocancel: Path
    clean_parquet: Path
    cleaning_summary_md: Path
    eda_findings_md: Path
    x_train_label: Path
    x_test_label: Path
    y_train: Path
    y_test: Path
    arr_delay_y_train: Path
    arr_delay_y_test: Path
    x_train_onehot_norm: Path
    x_test_onehot_norm: Path
    x_train_lstm: Path
    x_test_lstm: Path
    y_train_lstm: Path
    y_test_lstm: Path
    airline_encoder: Path
    airport_encoder: Path
    standard_scaler: Path
    minmax_scaler: Path


@dataclass(frozen=True)
class SplitConfig:
    test_size: float


@dataclass(frozen=True)
class CleaningConfig:
    iqr_multiplier: float


@dataclass(frozen=True)
class Config:
    seed: int
    dataset: DatasetConfig
    paths: PathsConfig
    split: SplitConfig
    cleaning: CleaningConfig
    targets: list[str] = field(default_factory=list)
    arrival_delay_col: str = "ARR_DELAY"
    features: list[str] = field(default_factory=list)
    exclusion_flags: list[str] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    repo_root: Path = REPO_ROOT


def _resolve(repo_root: Path, relative: str) -> Path:
    return (repo_root / relative).resolve()


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load configs/config.yaml into a frozen Config with absolute paths."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    repo_root = REPO_ROOT
    raw_paths = raw["paths"]
    paths = PathsConfig(
        data_raw_dir=_resolve(repo_root, raw_paths["data_raw_dir"]),
        data_interim_dir=_resolve(repo_root, raw_paths["data_interim_dir"]),
        data_processed_dir=_resolve(repo_root, raw_paths["data_processed_dir"]),
        models_dir=_resolve(repo_root, raw_paths["models_dir"]),
        preprocessors_dir=_resolve(repo_root, raw_paths["preprocessors_dir"]),
        reports_dir=_resolve(repo_root, raw_paths["reports_dir"]),
        figures_dir=_resolve(repo_root, raw_paths["figures_dir"]),
        raw_csv=_resolve(repo_root, raw_paths["raw_csv"]),
        cancelled_diverted_monthly=_resolve(repo_root, raw_paths["cancelled_diverted_monthly"]),
        flights_nocancel=_resolve(repo_root, raw_paths["flights_nocancel"]),
        clean_parquet=_resolve(repo_root, raw_paths["clean_parquet"]),
        cleaning_summary_md=_resolve(repo_root, raw_paths["cleaning_summary_md"]),
        eda_findings_md=_resolve(repo_root, raw_paths["eda_findings_md"]),
        x_train_label=_resolve(repo_root, raw_paths["x_train_label"]),
        x_test_label=_resolve(repo_root, raw_paths["x_test_label"]),
        y_train=_resolve(repo_root, raw_paths["y_train"]),
        y_test=_resolve(repo_root, raw_paths["y_test"]),
        arr_delay_y_train=_resolve(repo_root, raw_paths["arr_delay_y_train"]),
        arr_delay_y_test=_resolve(repo_root, raw_paths["arr_delay_y_test"]),
        x_train_onehot_norm=_resolve(repo_root, raw_paths["x_train_onehot_norm"]),
        x_test_onehot_norm=_resolve(repo_root, raw_paths["x_test_onehot_norm"]),
        x_train_lstm=_resolve(repo_root, raw_paths["x_train_lstm"]),
        x_test_lstm=_resolve(repo_root, raw_paths["x_test_lstm"]),
        y_train_lstm=_resolve(repo_root, raw_paths["y_train_lstm"]),
        y_test_lstm=_resolve(repo_root, raw_paths["y_test_lstm"]),
        airline_encoder=_resolve(repo_root, raw_paths["airline_encoder"]),
        airport_encoder=_resolve(repo_root, raw_paths["airport_encoder"]),
        standard_scaler=_resolve(repo_root, raw_paths["standard_scaler"]),
        minmax_scaler=_resolve(repo_root, raw_paths["minmax_scaler"]),
    )

    config = Config(
        seed=raw["seed"],
        dataset=DatasetConfig(**raw["dataset"]),
        paths=paths,
        split=SplitConfig(**raw["split"]),
        cleaning=CleaningConfig(**raw["cleaning"]),
        targets=list(raw.get("targets", [])),
        arrival_delay_col=raw.get("arrival_delay_col", "ARR_DELAY"),
        features=list(raw.get("features", [])),
        exclusion_flags=list(raw.get("exclusion_flags", [])),
        models=dict(raw.get("models", {})),
        repo_root=repo_root,
    )
    logger.debug("Loaded config from %s", config_path)
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    logger.info("Repo root: %s", cfg.repo_root)
    logger.info("Raw CSV path: %s", cfg.paths.raw_csv)
    logger.info("Targets: %s", cfg.targets)
