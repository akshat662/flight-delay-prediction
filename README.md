# Flight Delay Prediction

## Problem statement

US domestic flights are frequently delayed, and delays are attributed to one
of five causes tracked by the Bureau of Transportation Statistics (BTS):
carrier, weather, National Airspace System (NAS), security, and late
aircraft. This project builds a reproducible, production-style pipeline to
predict:

- the five delay-component targets — `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`,
  `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`
- overall arrival delay (`ARR_DELAY`)

for US domestic flights, using historical BTS data.

## Dataset

[`patrickzel/flight-delay-and-cancellation-dataset-2019-2023`](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
on Kaggle — `flights_sample_3m.csv`, ~3M rows of US DOT BTS flight records,
August 2019 through August 2023. Downloaded programmatically via `kagglehub`.

## Repo layout

```
configs/config.yaml       All configuration: paths, seed, targets, features, hyperparameters
data/
  raw/                     Downloaded source CSV (gitignored)
  interim/                 Intermediate artifacts (gitignored)
  processed/               Modeling-ready Parquet (gitignored)
models/                    Trained model artifacts (gitignored)
reports/
  figures/                 Generated plots (gitignored)
  cleaning_summary.md      Row counts and stats from the cleaning run
notebooks/                 Exploratory notebooks
src/
  config.py                Loads configs/config.yaml into a frozen dataclass
  data/
    download.py             Kaggle dataset download
    clean.py                Cleaning pipeline
  utils/
    seed.py                 Deterministic seeding
    io.py                   Parquet save/load + cache helper
tests/                     Pytest unit tests
```

## Setup

```bash
make install
make data    # downloads the raw CSV and runs the cleaning pipeline
make test
```

Every module is also runnable directly, e.g. `python -m src.data.clean --force`.

## Pipeline status

- [x] Phase 1 — Scaffold, data acquisition, cleaning
- [ ] Phase 2 — EDA and figures
- [ ] Phase 3 — Feature-selection statistics (Pearson, Kruskal–Wallis, PCA check)
- [ ] Phase 4 — Feature engineering, encoding, splits, scaling
- [ ] Phase 5 — Baseline models (XGBoost, ANN)
- [ ] Phase 6 — Sequence models (LSTM, LSTM+CNN hybrid)
- [ ] Phase 7 — Evaluation, comparison table, README
- [ ] Phase 8 — Rescheduling optimizer (deferred)
