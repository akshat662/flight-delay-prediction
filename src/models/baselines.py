"""Baseline XGBoost and ANN models for the five-component delay target.

Ports reference/FINAL.ipynb cells 73 (imports), 76 (ANN), and 79-81 (XGBoost
+ its actual-vs-predicted scatter plots) — the two non-time-series baselines.
Inputs are data/processed/X_train_label.parquet / X_test_label.parquet /
Y_train.parquet / Y_test.parquet, produced by the feature-engineering
pipeline (src.features.build).

Notebook departures:
  - `set_all_seeds(42)` is called first in main(), before any data or model
    work. The notebook never seeds numpy/random/TensorFlow at all — only
    XGBRegressor's own `random_state=42` — so its ANN runs are not
    reproducible; this fixes that.
  - A training-mean-predictor floor and the ANN loss-curve plot are not in
    the notebook; both are added here.
  - The notebook never persists either model; this module adds
    `save_model`/`model.save` calls.
  - Metrics are written to JSON (reports/metrics/) rather than only printed.

Run as: python -m src.models.baselines --model {xgb,ann,all}
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

from src.config import Config, load_config
from src.utils.io import load_df
from src.utils.plotting import new_figure, save_and_close, style_axes
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

# Reference numbers from the source notebook's report, for comparison only.
REFERENCE_METRICS = {
    "xgboost": {"mse": 345.02, "mae": 9.53},
    "ann": {"mse": 1118.49, "mae": 10.02},
}


def compute_metrics(y_true: pd.DataFrame, y_pred: np.ndarray, target_cols: list[str]) -> dict:
    """Overall MSE/MAE plus per-component MAE (multioutput='raw_values')."""
    mae_per_component = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_per_component": {col: float(v) for col, v in zip(target_cols, mae_per_component)},
    }


def mean_predictor_floor(
    Y_train: pd.DataFrame, Y_test: pd.DataFrame, target_cols: list[str]
) -> dict:
    """Predict each component's training mean for every row; report MSE/MAE.

    The only way to tell whether a model has learned anything beyond this
    floor, given every feature correlates below r=0.08 with the target.
    """
    train_mean = Y_train.mean()
    y_pred_train = np.tile(train_mean.to_numpy(), (len(Y_train), 1))
    y_pred_test = np.tile(train_mean.to_numpy(), (len(Y_test), 1))

    return {
        "train": compute_metrics(Y_train, y_pred_train, target_cols),
        "test": compute_metrics(Y_test, y_pred_test, target_cols),
    }


def _scatter_actual_vs_predicted(
    y_true: pd.DataFrame, y_pred: np.ndarray, name: str, title: str, config: Config
) -> None:
    fig, ax = new_figure((8, 8))
    ax.scatter(y_true, y_pred, s=2, alpha=0.1)
    style_axes(ax, title, xlabel="Actual Values", ylabel="Predicted Values")
    save_and_close(fig, name, config)


def train_xgb(
    X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame,
    target_cols: list[str], config: Config,
) -> dict:
    """cell 79: XGBRegressor(objective='reg:squarederror', random_state=42), fit, predict, metrics."""
    xgb_params = dict(config.models.get("xgboost", {}))
    xgb_regressor = XGBRegressor(random_state=config.seed, **xgb_params)
    xgb_regressor.fit(X_train, Y_train)

    Y_train_pred = xgb_regressor.predict(X_train)
    Y_test_pred = xgb_regressor.predict(X_test)

    metrics = {
        "train": compute_metrics(Y_train, Y_train_pred, target_cols),
        "test": compute_metrics(Y_test, Y_test_pred, target_cols),
    }
    logger.info(
        "XGBoost: train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        metrics["train"]["mse"], metrics["train"]["mae"],
        metrics["test"]["mse"], metrics["test"]["mae"],
    )
    logger.info(
        "XGBoost reference (source report): mse=%.2f mae=%.2f",
        REFERENCE_METRICS["xgboost"]["mse"], REFERENCE_METRICS["xgboost"]["mae"],
    )

    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    xgb_regressor.save_model(config.paths.xgb_baseline_model)
    logger.info("Saved XGBoost model to %s", config.paths.xgb_baseline_model)

    # cells 80-81: actual-vs-predicted scatter plots (train, then test)
    _scatter_actual_vs_predicted(
        Y_train, Y_train_pred, "scatter_xgb_train_actual_vs_predicted", "XGBoost: Actual vs Predicted (Train)", config
    )
    _scatter_actual_vs_predicted(
        Y_test, Y_test_pred, "scatter_xgb_test_actual_vs_predicted", "XGBoost: Actual vs Predicted (Test)", config
    )

    return metrics


def build_ann_model(n_features: int, n_targets: int, ann_cfg: dict) -> Sequential:
    """cell 76: Dense(64,relu) -> Dense(32,relu) -> Dense(5), adam/mse/mae."""
    hidden_layers = ann_cfg.get("hidden_layers", [64, 32])
    activation = ann_cfg.get("activation", "relu")
    optimizer = ann_cfg.get("optimizer", "adam")
    loss = ann_cfg.get("loss", "mse")
    metric_names = ann_cfg.get("metrics", ["mae"])

    model = Sequential(
        [Dense(hidden_layers[0], activation=activation, input_shape=(n_features,))]
        + [Dense(n, activation=activation) for n in hidden_layers[1:]]
        + [Dense(n_targets)]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_names)
    return model


def train_ann(
    X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame,
    target_cols: list[str], config: Config,
) -> dict:
    """cell 76: build, train, and evaluate the ANN; 50 epochs, batch 32."""
    ann_cfg = config.models.get("ann", {})
    epochs = ann_cfg.get("epochs", 50)
    batch_size = ann_cfg.get("batch_size", 32)
    validation_split = ann_cfg.get("validation_split", 0.2)

    model = build_ann_model(X_train.shape[1], len(target_cols), ann_cfg)

    history = model.fit(
        X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=2
    )

    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
    logger.info("ANN evaluate(): test loss=%.4f, test mae=%.4f", test_loss, test_mae)

    Y_train_pred = model.predict(X_train, verbose=0)
    Y_test_pred = model.predict(X_test, verbose=0)

    metrics = {
        "train": compute_metrics(Y_train, Y_train_pred, target_cols),
        "test": compute_metrics(Y_test, Y_test_pred, target_cols),
    }
    logger.info(
        "ANN: train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        metrics["train"]["mse"], metrics["train"]["mae"],
        metrics["test"]["mse"], metrics["test"]["mae"],
    )
    logger.info(
        "ANN reference (source report): mse=%.2f mae=%.2f",
        REFERENCE_METRICS["ann"]["mse"], REFERENCE_METRICS["ann"]["mae"],
    )

    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.paths.ann_baseline_model)
    logger.info("Saved ANN model to %s", config.paths.ann_baseline_model)

    _plot_loss_curve(history, config)

    return metrics


def _plot_loss_curve(history, config: Config) -> None:
    fig, ax = new_figure()
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    style_axes(ax, "ANN Training Loss", xlabel="Epoch", ylabel="MSE")
    ax.legend()
    save_and_close(fig, "ann_loss_curve", config)


def _load_existing_metrics(config: Config) -> dict:
    if config.paths.baseline_metrics_json.exists():
        with config.paths.baseline_metrics_json.open("r") as f:
            return json.load(f)
    return {}


def _write_metrics(metrics: dict, config: Config) -> None:
    config.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    with config.paths.baseline_metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", config.paths.baseline_metrics_json)


def main(model_choice: str = "all") -> None:
    set_all_seeds(42)

    config = load_config()
    target_cols = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_SECURITY", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]

    X_train = load_df(config.paths.x_train_label)
    X_test = load_df(config.paths.x_test_label)
    Y_train = load_df(config.paths.y_train)
    Y_test = load_df(config.paths.y_test)

    metrics = _load_existing_metrics(config)

    floor = mean_predictor_floor(Y_train, Y_test, target_cols)
    metrics["mean_predictor_floor"] = floor
    logger.info(
        "Mean-predictor floor: train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        floor["train"]["mse"], floor["train"]["mae"], floor["test"]["mse"], floor["test"]["mae"],
    )

    if model_choice in ("xgb", "all"):
        metrics["xgboost"] = train_xgb(X_train, Y_train, X_test, Y_test, target_cols, config)

    if model_choice in ("ann", "all"):
        metrics["ann"] = train_ann(X_train, Y_train, X_test, Y_test, target_cols, config)

    metrics["reference"] = REFERENCE_METRICS
    _write_metrics(metrics, config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost/ANN models")
    parser.add_argument("--model", choices=["xgb", "ann", "all"], default="all", help="Which model(s) to train")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    main(model_choice=args.model)
