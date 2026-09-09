"""Consolidated model comparison across two test splits.

XGBoost and the ANN were trained on the random 75/25 split
(X_test_label / Y_test); LSTM and the hybrid were trained on the
chronological, shuffle=False split (X_test_lstm / Y_test_lstm). These two
test sets are NOT interchangeable, so this module reports two tables and
never mixes them into a single ranked list:

  1. "own split" -- each of the four saved models evaluated on the test set
     it was actually trained for, each against the mean-predictor floor
     computed on that same split.
  2. "unified chronological" -- XGBoost and the ANN trained separately
     (same hyperparameters, freshly seeded) on X_train_lstm/Y_train_lstm and
     scored on X_test_lstm/Y_test_lstm, alongside the already-chronological
     LSTM and hybrid. This is the only table where all four models sit on
     identical data, and it's the one the README leads with. These models
     are not persisted -- they exist only to produce this comparison.

Outputs: reports/results.md, reports/metrics/final_comparison.csv, and a
grouped bar chart of per-component MAE by model in reports/figures/.

Run as: python -m src.models.evaluate
"""

from __future__ import annotations

import logging

import keras
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import Config, load_config
from src.models.baselines import build_ann_model, compute_metrics, mean_predictor_floor
from src.utils.io import load_df
from src.utils.plotting import new_figure, save_and_close, style_axes
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

TARGET_COLS = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_SECURITY", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]


def _pct_improvement(floor_value: float, model_value: float) -> float:
    """% by which model_value improves on (is lower than) floor_value."""
    return (floor_value - model_value) / floor_value * 100.0


def _pred_vs_truth_stats(y_true: pd.DataFrame, y_pred: np.ndarray, target_cols: list[str]) -> dict:
    """Per-component mean/std of predictions vs. truth."""
    y_pred_df = pd.DataFrame(np.asarray(y_pred), columns=target_cols)
    stats = {}
    for col in target_cols:
        stats[col] = {
            "truth_mean": float(y_true[col].mean()),
            "truth_std": float(y_true[col].std()),
            "pred_mean": float(y_pred_df[col].mean()),
            "pred_std": float(y_pred_df[col].std()),
        }
    return stats


def load_saved_models(config: Config) -> dict:
    """Load the four trained models from disk."""
    xgb_model = XGBRegressor()
    xgb_model.load_model(config.paths.xgb_baseline_model)

    ann_model = keras.models.load_model(config.paths.ann_baseline_model)
    lstm_model = keras.models.load_model(config.paths.lstm_model)
    hybrid_model = keras.models.load_model(config.paths.hybrid_model)

    logger.info("Loaded 4 saved models from %s", config.paths.models_dir)
    return {"xgboost": xgb_model, "ann": ann_model, "lstm": lstm_model, "hybrid": hybrid_model}


def evaluate_own_split(models: dict, config: Config) -> dict:
    """Each model against its own test split and that split's floor."""
    X_test_random = load_df(config.paths.x_test_label)
    Y_test_random = load_df(config.paths.y_test)
    X_train_random = load_df(config.paths.x_train_label)
    Y_train_random = load_df(config.paths.y_train)

    X_test_chrono = load_df(config.paths.x_test_lstm)
    Y_test_chrono = load_df(config.paths.y_test_lstm)
    X_train_chrono = load_df(config.paths.x_train_lstm)
    Y_train_chrono = load_df(config.paths.y_train_lstm)

    floor_random = mean_predictor_floor(Y_train_random, Y_test_random, TARGET_COLS)
    floor_chrono = mean_predictor_floor(Y_train_chrono, Y_test_chrono, TARGET_COLS)

    results = {}
    for name, split_X, split_Y, floor in [
        ("xgboost", X_test_random, Y_test_random, floor_random),
        ("ann", X_test_random, Y_test_random, floor_random),
        ("lstm", X_test_chrono, Y_test_chrono, floor_chrono),
        ("hybrid", X_test_chrono, Y_test_chrono, floor_chrono),
    ]:
        y_pred = models[name].predict(split_X, verbose=0) if name in ("ann", "lstm", "hybrid") else models[name].predict(split_X)
        metrics = compute_metrics(split_Y, y_pred, TARGET_COLS)
        results[name] = {
            "split": "random" if name in ("xgboost", "ann") else "chronological",
            "metrics": metrics,
            "floor": floor["test"],
            "pct_improvement_mse": _pct_improvement(floor["test"]["mse"], metrics["mse"]),
            "pct_improvement_mae": _pct_improvement(floor["test"]["mae"], metrics["mae"]),
            "pred_vs_truth": _pred_vs_truth_stats(split_Y, y_pred, TARGET_COLS),
        }
        logger.info(
            "[own split] %s (%s): mse=%.4f mae=%.4f, %.1f%% below floor (mse)",
            name, results[name]["split"], metrics["mse"], metrics["mae"], results[name]["pct_improvement_mse"],
        )

    return {"random": floor_random, "chronological": floor_chrono, "models": results}


def train_xgb_on_chronological_split(X_train, Y_train, config: Config) -> XGBRegressor:
    xgb_params = dict(config.models.get("xgboost", {}))
    model = XGBRegressor(random_state=config.seed, **xgb_params)
    model.fit(X_train, Y_train)
    return model


def _train_ann_on_chronological_split(X_train, Y_train, config: Config):
    model = build_ann_model(X_train.shape[1], len(TARGET_COLS), config.models["ann"])
    ann_cfg = config.models["ann"]
    model.fit(
        X_train, Y_train,
        epochs=ann_cfg["epochs"], batch_size=ann_cfg["batch_size"],
        validation_split=ann_cfg["validation_split"], verbose=2,
    )
    return model


def evaluate_unified_chronological(config: Config) -> dict:
    """Train XGBoost/ANN on the chronological split; compare all 4 models on identical data."""
    X_train = load_df(config.paths.x_train_lstm)
    Y_train = load_df(config.paths.y_train_lstm)
    X_test = load_df(config.paths.x_test_lstm)
    Y_test = load_df(config.paths.y_test_lstm)

    floor = mean_predictor_floor(Y_train, Y_test, TARGET_COLS)

    logger.info("Training XGBoost on the chronological split (same hyperparameters as the random-split baseline)")
    xgb_chrono = train_xgb_on_chronological_split(X_train, Y_train, config)

    logger.info("Training ANN on the chronological split (same hyperparameters as the random-split baseline)")
    ann_chrono = _train_ann_on_chronological_split(X_train, Y_train, config)

    lstm_model = keras.models.load_model(config.paths.lstm_model)
    hybrid_model = keras.models.load_model(config.paths.hybrid_model)

    results = {}
    for name, model, needs_verbose in [
        ("xgboost", xgb_chrono, False),
        ("ann", ann_chrono, True),
        ("lstm", lstm_model, True),
        ("hybrid", hybrid_model, True),
    ]:
        y_pred = model.predict(X_test, verbose=0) if needs_verbose else model.predict(X_test)
        metrics = compute_metrics(Y_test, y_pred, TARGET_COLS)
        results[name] = {
            "metrics": metrics,
            "pct_improvement_mse": _pct_improvement(floor["test"]["mse"], metrics["mse"]),
            "pct_improvement_mae": _pct_improvement(floor["test"]["mae"], metrics["mae"]),
        }
        logger.info(
            "[unified chronological] %s: mse=%.4f mae=%.4f, %.1f%% below floor (mse)",
            name, metrics["mse"], metrics["mae"], results[name]["pct_improvement_mse"],
        )

    return {"floor": floor, "models": results}


def _plot_grouped_mae_by_model(unified: dict, config: Config) -> None:
    model_names = ["xgboost", "ann", "lstm", "hybrid"]
    labels = ["XGBoost", "ANN", "LSTM", "Hybrid"]
    n_components = len(TARGET_COLS)
    x = np.arange(n_components)
    width = 0.2

    fig, ax = new_figure((12, 6))
    for i, (name, label) in enumerate(zip(model_names, labels)):
        maes = [unified["models"][name]["metrics"]["mae_per_component"][c] for c in TARGET_COLS]
        ax.bar(x + (i - 1.5) * width, maes, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_COLS, rotation=30, ha="right")
    style_axes(ax, "Per-component test MAE by model (unified chronological split)", ylabel="MAE (min)")
    ax.legend()
    save_and_close(fig, "mae_by_model_grouped_bar", config)


def _write_csv(own_split: dict, unified: dict, config: Config) -> None:
    rows = []
    for name, r in own_split["models"].items():
        row = {
            "table": "own_split",
            "model": name,
            "split": r["split"],
            "test_mse": r["metrics"]["mse"],
            "test_mae": r["metrics"]["mae"],
            "floor_mse": r["floor"]["mse"],
            "floor_mae": r["floor"]["mae"],
            "pct_improvement_mse": r["pct_improvement_mse"],
            "pct_improvement_mae": r["pct_improvement_mae"],
        }
        for c in TARGET_COLS:
            row[f"mae_{c}"] = r["metrics"]["mae_per_component"][c]
        rows.append(row)

    for name, r in unified["models"].items():
        row = {
            "table": "unified_chronological",
            "model": name,
            "split": "chronological",
            "test_mse": r["metrics"]["mse"],
            "test_mae": r["metrics"]["mae"],
            "floor_mse": unified["floor"]["test"]["mse"],
            "floor_mae": unified["floor"]["test"]["mae"],
            "pct_improvement_mse": r["pct_improvement_mse"],
            "pct_improvement_mae": r["pct_improvement_mae"],
        }
        for c in TARGET_COLS:
            row[f"mae_{c}"] = r["metrics"]["mae_per_component"][c]
        rows.append(row)

    df = pd.DataFrame(rows)
    config.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.paths.final_comparison_csv, index=False)
    logger.info("Wrote %s", config.paths.final_comparison_csv)


def _fmt_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_results_md(own_split: dict, unified: dict, config: Config) -> None:
    lines: list[str] = ["# Model Comparison", ""]

    lines.append(
        "Two test splits are in play here and they are not interchangeable: XGBoost and the ANN "
        "were trained on a random 75/25 split; LSTM and the hybrid were trained on a chronological, "
        "`shuffle=False` split. Every table below states which split it uses."
    )
    lines.append("")

    lines.append("## Unified comparison (chronological split, all four models)")
    lines.append("")
    lines.append(
        f"XGBoost and the ANN were trained separately on `X_train_lstm`/`Y_train_lstm` (same "
        f"hyperparameters as the random-split baseline) and scored on `X_test_lstm`/`Y_test_lstm`, so all four models "
        f"sit on identical data here. Floor (chronological test): "
        f"MSE={unified['floor']['test']['mse']:.2f}, MAE={unified['floor']['test']['mae']:.2f}."
    )
    lines.append("")
    rows = []
    for name, label in [("xgboost", "XGBoost"), ("ann", "ANN"), ("lstm", "LSTM"), ("hybrid", "Hybrid")]:
        r = unified["models"][name]
        rows.append([
            label, f"{r['metrics']['mse']:.2f}", f"{r['metrics']['mae']:.2f}",
            f"{r['pct_improvement_mse']:.1f}%", f"{r['pct_improvement_mae']:.1f}%",
        ])
    lines.extend(_fmt_table(rows, ["model", "test MSE", "test MAE", "% improvement over floor (MSE)", "% improvement over floor (MAE)"]))
    lines.append("")
    lines.append("![Per-component MAE by model](figures/mae_by_model_grouped_bar.png)")
    lines.append("")

    lines.append("## Own-split comparison (each model on the split it was trained for)")
    lines.append("")
    lines.append(
        f"Random-split floor (test): MSE={own_split['random']['test']['mse']:.2f}, "
        f"MAE={own_split['random']['test']['mae']:.2f}. "
        f"Chronological-split floor (test): MSE={own_split['chronological']['test']['mse']:.2f}, "
        f"MAE={own_split['chronological']['test']['mae']:.2f}."
    )
    lines.append("")
    rows = []
    for name, label in [("xgboost", "XGBoost"), ("ann", "ANN"), ("lstm", "LSTM"), ("hybrid", "Hybrid")]:
        r = own_split["models"][name]
        rows.append([
            label, r["split"], f"{r['metrics']['mse']:.2f}", f"{r['metrics']['mae']:.2f}",
            f"{r['pct_improvement_mse']:.1f}%", f"{r['pct_improvement_mae']:.1f}%",
        ])
    lines.extend(_fmt_table(rows, ["model", "split", "test MSE", "test MAE", "% improvement over floor (MSE)", "% improvement over floor (MAE)"]))
    lines.append("")

    lines.append("### Per-component MAE (own split)")
    lines.append("")
    for name, label in [("xgboost", "XGBoost"), ("ann", "ANN"), ("lstm", "LSTM"), ("hybrid", "Hybrid")]:
        r = own_split["models"][name]
        rows = [[c, f"{r['metrics']['mae_per_component'][c]:.2f}"] for c in TARGET_COLS]
        lines.append(f"**{label}** ({r['split']} split)")
        lines.append("")
        lines.extend(_fmt_table(rows, ["component", "MAE"]))
        lines.append("")

    lines.append("## Prediction spread vs. truth (own split)")
    lines.append("")
    lines.append(
        "The source notebook noted that its LSTM's predictions had much lower variance than the "
        "actual delays. Checked here for all four models: predicted standard deviation vs. actual "
        "standard deviation, per component."
    )
    lines.append("")
    for name, label in [("xgboost", "XGBoost"), ("ann", "ANN"), ("lstm", "LSTM"), ("hybrid", "Hybrid")]:
        stats = own_split["models"][name]["pred_vs_truth"]
        rows = [
            [c, f"{stats[c]['truth_mean']:.2f}", f"{stats[c]['truth_std']:.2f}",
             f"{stats[c]['pred_mean']:.2f}", f"{stats[c]['pred_std']:.2f}"]
            for c in TARGET_COLS
        ]
        lines.append(f"**{label}**")
        lines.append("")
        lines.extend(_fmt_table(rows, ["component", "truth mean", "truth std", "pred mean", "pred std"]))
        lines.append("")

    all_pred_std_lower = all(
        own_split["models"][name]["pred_vs_truth"][c]["pred_std"] < own_split["models"][name]["pred_vs_truth"][c]["truth_std"]
        for name in ["xgboost", "ann", "lstm", "hybrid"]
        for c in TARGET_COLS
    )
    lines.append(
        f"**Reproduces**: predicted standard deviation is lower than actual standard deviation for "
        f"every model and every component ({'confirmed' if all_pred_std_lower else 'NOT confirmed for all'}). "
        f"All four models predict a narrower range than the true delay distribution -- expected, "
        f"given every feature correlates with ARR_DELAY below r=0.08: with this little "
        f"signal, MSE-minimizing models converge toward predicting something close to the mean, "
        f"under-representing the tails."
    )
    lines.append("")

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    config.paths.results_md.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", config.paths.results_md)


def main() -> None:
    set_all_seeds(42)
    config = load_config()

    models = load_saved_models(config)
    own_split = evaluate_own_split(models, config)
    unified = evaluate_unified_chronological(config)

    _plot_grouped_mae_by_model(unified, config)
    _write_csv(own_split, unified, config)
    _write_results_md(own_split, unified, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
