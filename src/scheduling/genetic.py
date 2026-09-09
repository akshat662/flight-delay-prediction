"""Flight rescheduling via a genetic algorithm.

Ports reference/FINAL.ipynb's GA section (cells 113-118): per-day search
over CRS_DEP_TIME using tournament selection, single-point crossover, and
bounded mutation, with the same constants (POPULATION_SIZE=50,
NUM_GENERATIONS=100, MUTATION_RATE=0.1, TOURNAMENT_SIZE=5) and the same
operators (init perturbation +/-30, mutation perturbation +/-63, both
clipped to [0, 2359]).

Two things are NOT ported as-is, per explicit instruction:

1. Fitness function. The notebook's `calculate_total_delay` reassigns
   `data_day['CRS_DEP_TIME']` to the candidate schedule but then sums the
   *unmodified* `DEP_DELAY` column -- it never recomputes delay from the
   candidate schedule. Its own logged run confirms this: every generation
   reports an identical "Best Score (Total Delay): 18974.0", proving the
   fitness never responded to the chromosome. Per the Phase 8 brief, this
   module instead scores a candidate schedule with the best model from
   Phase 7 (XGBoost retrained on the chronological split): each flight's
   CRS_DEP_TIME is swapped for its candidate value, the model predicts the
   five DELAY_DUE_* components under that schedule, and the fitness is
   their sum across the day's flights. This changes *what* is minimized
   (model-predicted component-sum delay, i.e. an ARR_DELAY proxy, instead
   of historical DEP_DELAY) because no model in this project predicts
   DEP_DELAY -- it does not change the GA's structure (still: aggregate a
   per-flight delay measure across the day, lower is better).

2. The "original" CRS_DEP_TIME in the notebook's own final output table is
   corrupted: the same in-place mutation above means that by the time cell
   118 runs, `data_day['CRS_DEP_TIME']` holds leftover, un-normalized
   chromosome values from whichever individual was scored last -- not the
   true schedule. Verified directly: the notebook's table shows 1993 and
   1078 for two flights whose actual scheduled times are 2050 and 1030.
   This module never mutates the source frame; `CRS_DEP_TIME` in the output
   is the untouched original, and only `Optimized_CRS_DEP_TIME` goes
   through the notebook's own `adjust_time_format` (cell 117).

Note on the GA's internal arithmetic (ported unchanged, not fixed): adding
a raw integer to an HHMM-encoded value is not equivalent to a same-sized
shift in real minutes whenever it changes the hundreds digit (e.g. 950 + 63
= 1013, which decodes as 10:13, not the correct 10:53). This is a
pre-existing property of the notebook's operators; per "same operators,
same constraints, do not redesign," it's preserved as-is and only cleaned
up for display via `adjust_time_format`, exactly as the notebook does.

Run as: python -m src.scheduling.genetic
"""

from __future__ import annotations

import logging
import random

import joblib
import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.features.build import convert_hhmm_to_mins
from src.models.evaluate import retrain_xgb_on_chronological_split
from src.utils.io import load_df
from src.utils.plotting import new_figure, save_and_close, style_axes
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

TARGET_COLS = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_SECURITY", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]
FEATURE_COLS = ["YEAR", "MONTH", "DAY", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "TAXI_IN", "TAXI_OUT"]
NUMERIC_COLS = ["CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "TAXI_IN", "TAXI_OUT"]


def adjust_time_format(time_array: np.ndarray) -> np.ndarray:
    """cell 117: carry excess minutes (>59) into hours, unchanged from the notebook."""
    adjusted_times = []
    for time in time_array:
        hh = time // 100
        mm = time % 100
        if mm > 59:
            hh += 1
            mm -= 60
        adjusted_times.append(hh * 100 + mm)
    return np.array(adjusted_times)


def initialize_population(original_times: np.ndarray, population_size: int, init_perturbation: int, time_bounds: tuple[int, int]) -> list[np.ndarray]:
    """cell 115: original schedule +/- init_perturbation, clipped to time_bounds."""
    population = []
    for _ in range(population_size):
        individual = original_times + np.random.randint(-init_perturbation, init_perturbation, size=len(original_times))
        individual = np.clip(individual, time_bounds[0], time_bounds[1])
        population.append(individual)
    return population


def tournament_selection(population: list[np.ndarray], scores: list[float], tournament_size: int) -> np.ndarray:
    """cell 115: sample tournament_size individuals, keep the lowest-scoring (best) one."""
    tournament = random.sample(list(zip(population, scores)), tournament_size)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cell 115: single-point crossover."""
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(individual: np.ndarray, mutation_rate: float, mutation_perturbation: int, time_bounds: tuple[int, int]) -> np.ndarray:
    """cell 115: per-gene mutation by +/-mutation_perturbation, clipped to time_bounds."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.randint(-mutation_perturbation, mutation_perturbation)
            individual[i] = np.clip(individual[i], time_bounds[0], time_bounds[1])
    return individual


def build_candidate_features(
    day_df: pd.DataFrame, crs_dep_times: np.ndarray, airline_encoder, airport_encoder, scaler
) -> pd.DataFrame:
    """Model-ready features for day_df's flights under a candidate CRS_DEP_TIME schedule.

    Mirrors src.features.build's preprocessing (label encoding, hhmm->minutes,
    Z-scaling) without mutating day_df -- unlike the notebook's fitness
    function, this is a pure function of its arguments.
    """
    features = pd.DataFrame(
        {
            "YEAR": day_df["FL_DATE"].dt.year.to_numpy(),
            "MONTH": day_df["FL_DATE"].dt.month.to_numpy(),
            "DAY": day_df["FL_DATE"].dt.day.to_numpy(),
            "AIRLINE": airline_encoder.transform(day_df["AIRLINE"]),
            "ORIGIN": airport_encoder.transform(day_df["ORIGIN"]),
            "DEST": airport_encoder.transform(day_df["DEST"]),
            "CRS_DEP_TIME": convert_hhmm_to_mins(crs_dep_times.astype(int).astype(str)),
            "CRS_ARR_TIME": convert_hhmm_to_mins(day_df["CRS_ARR_TIME"].astype(int).astype(str)),
            "DISTANCE": day_df["DISTANCE"].to_numpy(),
            "TAXI_IN": day_df["TAXI_IN"].to_numpy(),
            "TAXI_OUT": day_df["TAXI_OUT"].to_numpy(),
        }
    )
    features[NUMERIC_COLS] = scaler.transform(features[NUMERIC_COLS])
    return features[FEATURE_COLS]


def calculate_total_delay(
    crs_dep_times: np.ndarray, day_df: pd.DataFrame, model, airline_encoder, airport_encoder, scaler
) -> float:
    """Fitness: total model-predicted component-sum delay for the day under this candidate schedule."""
    features = build_candidate_features(day_df, crs_dep_times, airline_encoder, airport_encoder, scaler)
    predicted_components = model.predict(features)
    return float(np.asarray(predicted_components).sum())


def genetic_algorithm(
    day_df: pd.DataFrame, model, airline_encoder, airport_encoder, scaler, ga_cfg: dict
) -> tuple[np.ndarray, float, list[float]]:
    """cell 116: the generational loop, unchanged in structure."""
    population_size = ga_cfg["population_size"]
    num_generations = ga_cfg["num_generations"]
    mutation_rate = ga_cfg["mutation_rate"]
    tournament_size = ga_cfg["tournament_size"]
    init_perturbation = ga_cfg["init_perturbation"]
    mutation_perturbation = ga_cfg["mutation_perturbation"]
    time_bounds = tuple(ga_cfg["time_bounds"])

    original_times = day_df["CRS_DEP_TIME"].to_numpy()
    population = initialize_population(original_times, population_size, init_perturbation, time_bounds)

    def score(individual: np.ndarray) -> float:
        return calculate_total_delay(individual, day_df, model, airline_encoder, airport_encoder, scaler)

    history: list[float] = []
    for generation in range(num_generations):
        scores = [score(ind) for ind in population]
        next_generation = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate, mutation_perturbation, time_bounds))
            next_generation.append(mutate(child2, mutation_rate, mutation_perturbation, time_bounds))

        population = next_generation
        best_score = min(scores)
        history.append(best_score)
        logger.info("Generation %d, Best Score (Total Delay): %.2f", generation + 1, best_score)

    final_scores = [score(ind) for ind in population]
    best_individual = population[int(np.argmin(final_scores))]
    best_score = min(final_scores)
    history.append(best_score)

    return best_individual, best_score, history


def _plot_convergence(history: list[float], config: Config) -> None:
    fig, ax = new_figure()
    ax.plot(range(1, len(history) + 1), history, marker=".")
    style_axes(ax, "GA Convergence: Best Fitness per Generation", xlabel="Generation", ylabel="Total predicted delay (min)")
    save_and_close(fig, "ga_convergence", config)


def run_rescheduler(
    day_df: pd.DataFrame, model, airline_encoder, airport_encoder, scaler, ga_cfg: dict
) -> tuple[pd.DataFrame, float, float, list[float]]:
    """Run the GA over one day's flights.

    Returns (output_table, baseline_score, best_score, history). Both scores
    are the same model-predicted-delay measure, so they're directly
    comparable: baseline_score is the day's original schedule scored the
    same way the GA scores every candidate.
    """
    original_times = day_df["CRS_DEP_TIME"].to_numpy()
    baseline_score = calculate_total_delay(original_times, day_df, model, airline_encoder, airport_encoder, scaler)

    best_individual, best_score, history = genetic_algorithm(day_df, model, airline_encoder, airport_encoder, scaler, ga_cfg)
    optimized_times = adjust_time_format(best_individual)

    output_table = day_df[["FL_DATE", "AIRLINE", "FL_NUMBER", "ORIGIN", "DEST", "CRS_DEP_TIME"]].copy()
    output_table["Optimized_CRS_DEP_TIME"] = optimized_times

    return output_table, baseline_score, best_score, history


def main() -> None:
    set_all_seeds(42)
    random.seed(42)

    config = load_config()
    ga_cfg = config.scheduling["genetic"]
    target_date = ga_cfg["target_date"]

    df = load_df(config.paths.clean_parquet)
    day_df = df[df["FL_DATE"] == target_date].copy().reset_index(drop=True)
    logger.info("Rescheduling %d flights on %s (from flights_clean.parquet)", len(day_df), target_date)

    airline_encoder = joblib.load(config.paths.airline_encoder)
    airport_encoder = joblib.load(config.paths.airport_encoder)
    scaler = joblib.load(config.paths.standard_scaler)

    X_train_chrono = load_df(config.paths.x_train_lstm)
    Y_train_chrono = load_df(config.paths.y_train_lstm)
    logger.info("Retraining XGBoost on the chronological split (Phase 7's model, not persisted to disk)")
    model = retrain_xgb_on_chronological_split(X_train_chrono, Y_train_chrono, config)

    output_table, baseline_score, best_score, history = run_rescheduler(
        day_df, model, airline_encoder, airport_encoder, scaler, ga_cfg
    )

    pct_change = (baseline_score - best_score) / baseline_score * 100.0 if baseline_score else float("nan")
    logger.info(
        "Baseline predicted total delay: %.2f min | Optimized: %.2f min | Change: %.2f%%",
        baseline_score, best_score, pct_change,
    )
    if best_score < baseline_score:
        logger.info("The optimized schedule reduces model-predicted total delay by %.2f%%.", pct_change)
    else:
        logger.warning(
            "The optimized schedule does NOT reduce model-predicted total delay (%.2f%% change). "
            "Given every feature correlates with delay below r=0.08 (Phase 3), this is plausible: "
            "there may be too little schedule-sensitive signal in the model for the GA to exploit.",
            pct_change,
        )

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(config.paths.rescheduling_results_csv, index=False)
    logger.info("Wrote %s", config.paths.rescheduling_results_csv)

    _plot_convergence(history, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
