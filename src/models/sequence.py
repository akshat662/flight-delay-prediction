"""Sequence models (LSTM, LSTM+CNN hybrid) for the five-component delay target.

Ports reference/FINAL.ipynb cells 88-99 (LSTM) and 92-93/107-110 (hybrid) —
the two time-series baselines. Inputs are data/processed/X_train_lstm.parquet
/ X_test_lstm.parquet / Y_train_lstm.parquet / Y_test_lstm.parquet from
Phase 4, which are chronologically ordered and split with shuffle=False.

Two notebook bugs, fixed rather than ported (per the Phase 6 brief):
  1. Checkpoint paths were inconsistent: cell 91 writes "models/lstm_model
     .keras" and cell 93 writes "../models/hybrid_model.keras" — relative
     paths that resolve differently depending on the notebook's cwd. Both
     checkpoints here use config paths (models/lstm_model.keras and
     models/hybrid_model.keras), with the directory created first. This is
     the likely reason the source report shows LSTM and hybrid with an
     identical MSE of 367.868: if "../models" didn't exist relative to the
     notebook's cwd at cell 93's execution time, ModelCheckpoint would have
     silently failed to write there, and cell 107's
     `load_model("../models/hybrid_model.keras")` would have loaded a stale
     file left over from a previous run — quite possibly the LSTM model
     itself, if the two relative paths ever pointed at the same file.
  2. Seeds were set in cell 88, after `import tensorflow as tf`/`import
     keras` but *between* that and the data-loading/model cells — i.e. late,
     not "at the top of the entry point." `set_all_seeds(42)` is now called
     first in main(), before any data load or model construction.

Additionally: after `.fit()` with `ModelCheckpoint(save_best_only=True)` and
`EarlyStopping` (no `restore_best_weights`), the in-memory model at the end
of `.fit()` is *not* necessarily the best-val_loss model — early stopping
just halts training up to `patience` epochs after the best epoch, without
rolling weights back. The notebook's own cells 95/107 reload from the
checkpoint file specifically for this reason (to evaluate the *best*
checkpoint, not the final epoch). This module does the same: after fit(), it
reloads the just-written checkpoint before computing metrics, which is both
the notebook's apparent intent and the correct fix for bug #1 above (each
model now reloads a checkpoint it actually wrote, this session).

Run as: python -m src.models.sequence --model {lstm,hybrid,all}
"""

from __future__ import annotations

import argparse
import json
import logging

import keras
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Concatenate, Conv1D, Dense, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from src.config import Config, load_config
from src.models.baselines import compute_metrics, mean_predictor_floor
from src.utils.io import load_df
from src.utils.plotting import new_figure, save_and_close, style_axes
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

TARGET_COLS = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_SECURITY", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]

# Reference numbers from the source notebook's report, for comparison only.
REFERENCE_METRICS = {
    "lstm": {"mse": 367.87, "mae": 9.93},
    "hybrid": {"mse": 367.87, "mae": 10.09},
}


def _confirm_chronological_validation_split(X_train: pd.DataFrame, validation_split: float) -> None:
    """Confirm Keras's validation_split carves off a LATER chronological period.

    Keras takes the validation fraction from the tail of the arrays as given
    (before any shuffling), and X_train is already sorted chronologically by
    Phase 4's build.py. So the held-out validation set here should be a
    genuinely later period than the portion actually fit on -- correct for a
    time-series model, and worth confirming empirically rather than assuming.
    """
    dates = pd.to_datetime(dict(year=X_train["YEAR"], month=X_train["MONTH"], day=X_train["DAY"]))
    n_val = int(round(len(X_train) * validation_split))
    fit_part_max_date = dates.iloc[: len(X_train) - n_val].max()
    val_part_min_date = dates.iloc[len(X_train) - n_val :].min()
    is_chronological = bool(val_part_min_date >= fit_part_max_date)
    logger.info(
        "validation_split=%.2f check: fit-portion max date=%s, held-out validation min date=%s "
        "-> validation is a later period: %s",
        validation_split, fit_part_max_date.date(), val_part_min_date.date(), is_chronological,
    )
    assert is_chronological, "validation_split did not carve off a later chronological period"


def build_lstm_model(n_timesteps: int, n_targets: int, hparams: dict) -> Sequential:
    """cell 91: LSTM(64, relu, recurrent_dropout=0.2) -> Dense(5)."""
    model = Sequential(
        [
            Input(shape=(n_timesteps, 1)),
            LSTM(
                units=hparams["units"],
                activation=hparams["activation"],
                recurrent_dropout=hparams["recurrent_dropout"],
            ),
            Dense(n_targets),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams["learning_rate"]),
        loss=hparams["loss"],
        metrics=hparams["metrics"],
    )
    return model


def build_hybrid_model(n_timesteps: int, n_targets: int, hparams: dict) -> Model:
    """cell 93: Conv1D->MaxPool->Flatten->Dense branch concatenated with an LSTM->Dense branch."""
    input_layer = Input(shape=(n_timesteps, 1))

    conv_layer = Conv1D(
        filters=hparams["conv_filters"], kernel_size=hparams["kernel_size"], activation=hparams["conv_activation"]
    )(input_layer)
    maxpool_layer = MaxPooling1D(pool_size=hparams["pool_size"])(conv_layer)
    flatten_layer = Flatten()(maxpool_layer)
    dense_cnn = Dense(hparams["dense_cnn_units"], activation=hparams["conv_activation"])(flatten_layer)

    lstm_layer = LSTM(hparams["lstm_units"], activation=hparams["lstm_activation"])(input_layer)
    dense_lstm = Dense(hparams["dense_lstm_units"], activation=hparams["lstm_activation"])(lstm_layer)

    concatenated = Concatenate()([dense_cnn, dense_lstm])
    output_layer = Dense(n_targets)(concatenated)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=hparams["optimizer"], loss=hparams["loss"], metrics=hparams["metrics"])
    return model


def _fit_with_checkpoint_reload(
    model, X_train: pd.DataFrame, Y_train: pd.DataFrame, hparams: dict, checkpoint_path, config: Config
):
    """Fit with ModelCheckpoint(save_best_only=True) + EarlyStopping, then reload the best checkpoint."""
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=hparams["early_stopping_patience"], verbose=1)

    history = model.fit(
        X_train,
        Y_train,
        epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        validation_split=hparams["validation_split"],
        callbacks=[checkpoint, early_stopping],
        verbose=2,
    )

    best_model = keras.models.load_model(checkpoint_path)
    logger.info("Reloaded best checkpoint from %s", checkpoint_path)
    return best_model, history


def _plot_loss_curve(history, name: str, title: str, config: Config) -> None:
    fig, ax = new_figure()
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    style_axes(ax, title, xlabel="Epoch", ylabel="MSE")
    ax.legend()
    save_and_close(fig, name, config)


def run_lstm(
    X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame, config: Config
) -> dict:
    hparams = config.models["lstm"]
    model = build_lstm_model(X_train.shape[1], len(TARGET_COLS), hparams)
    best_model, history = _fit_with_checkpoint_reload(model, X_train, Y_train, hparams, config.paths.lstm_model, config)

    Y_train_pred = best_model.predict(X_train, verbose=0)
    Y_test_pred = best_model.predict(X_test, verbose=0)
    metrics = {
        "train": compute_metrics(Y_train, Y_train_pred, TARGET_COLS),
        "test": compute_metrics(Y_test, Y_test_pred, TARGET_COLS),
    }
    logger.info(
        "LSTM: train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        metrics["train"]["mse"], metrics["train"]["mae"], metrics["test"]["mse"], metrics["test"]["mae"],
    )
    logger.info(
        "LSTM reference (source report): mse=%.2f mae=%.2f",
        REFERENCE_METRICS["lstm"]["mse"], REFERENCE_METRICS["lstm"]["mae"],
    )

    _plot_loss_curve(history, "lstm_loss_curve", "LSTM Training Loss", config)
    return metrics


def run_hybrid(
    X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame, config: Config
) -> dict:
    hparams = config.models["lstm_cnn_hybrid"]
    model = build_hybrid_model(X_train.shape[1], len(TARGET_COLS), hparams)
    best_model, history = _fit_with_checkpoint_reload(
        model, X_train, Y_train, hparams, config.paths.hybrid_model, config
    )

    Y_train_pred = best_model.predict(X_train, verbose=0)
    Y_test_pred = best_model.predict(X_test, verbose=0)
    metrics = {
        "train": compute_metrics(Y_train, Y_train_pred, TARGET_COLS),
        "test": compute_metrics(Y_test, Y_test_pred, TARGET_COLS),
    }
    logger.info(
        "Hybrid: train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        metrics["train"]["mse"], metrics["train"]["mae"], metrics["test"]["mse"], metrics["test"]["mae"],
    )
    logger.info(
        "Hybrid reference (source report): mse=%.2f mae=%.2f",
        REFERENCE_METRICS["hybrid"]["mse"], REFERENCE_METRICS["hybrid"]["mae"],
    )

    _plot_loss_curve(history, "hybrid_loss_curve", "LSTM+CNN Hybrid Training Loss", config)
    return metrics


def _load_existing_metrics(config: Config) -> dict:
    if config.paths.sequence_metrics_json.exists():
        with config.paths.sequence_metrics_json.open("r") as f:
            return json.load(f)
    return {}


def _write_metrics(metrics: dict, config: Config) -> None:
    config.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    with config.paths.sequence_metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", config.paths.sequence_metrics_json)


def main(model_choice: str = "all") -> None:
    set_all_seeds(42)

    config = load_config()

    X_train = load_df(config.paths.x_train_lstm)
    X_test = load_df(config.paths.x_test_lstm)
    Y_train = load_df(config.paths.y_train_lstm)
    Y_test = load_df(config.paths.y_test_lstm)

    _confirm_chronological_validation_split(X_train, config.models["lstm"]["validation_split"])

    metrics = _load_existing_metrics(config)

    floor = mean_predictor_floor(Y_train, Y_test, TARGET_COLS)
    metrics["mean_predictor_floor"] = floor
    logger.info(
        "Mean-predictor floor (chronological test set): train mse=%.4f mae=%.4f | test mse=%.4f mae=%.4f",
        floor["train"]["mse"], floor["train"]["mae"], floor["test"]["mse"], floor["test"]["mae"],
    )

    if model_choice in ("lstm", "all"):
        metrics["lstm"] = run_lstm(X_train, Y_train, X_test, Y_test, config)

    if model_choice in ("hybrid", "all"):
        metrics["hybrid"] = run_hybrid(X_train, Y_train, X_test, Y_test, config)

    if "lstm" in metrics and "hybrid" in metrics:
        if metrics["lstm"]["test"]["mse"] == metrics["hybrid"]["test"]["mse"]:
            logger.warning(
                "LSTM and hybrid test MSE are IDENTICAL (%.6f) even after the checkpoint-path fix. "
                "This should not happen with two different architectures -- something else is shared "
                "(e.g. a stale checkpoint file, or predictions being computed from the same model object).",
                metrics["lstm"]["test"]["mse"],
            )

    metrics["reference"] = REFERENCE_METRICS
    _write_metrics(metrics, config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence models (LSTM, LSTM+CNN hybrid)")
    parser.add_argument("--model", choices=["lstm", "hybrid", "all"], default="all", help="Which model(s) to train")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    main(model_choice=args.model)
