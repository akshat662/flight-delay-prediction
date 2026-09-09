# Model Comparison

Two test splits are in play here and they are not interchangeable: XGBoost and the ANN were trained on a random 75/25 split; LSTM and the hybrid were trained on a chronological, `shuffle=False` split. Every table below states which split it uses.

## Unified comparison (chronological split, all four models)

XGBoost and the ANN were trained separately on `X_train_lstm`/`Y_train_lstm` (same hyperparameters as the random-split baseline) and scored on `X_test_lstm`/`Y_test_lstm`, so all four models sit on identical data here. Floor (chronological test): MSE=409.24, MAE=11.37.

| model | test MSE | test MAE | % improvement over floor (MSE) | % improvement over floor (MAE) |
|---|---|---|---|---|
| XGBoost | 347.87 | 9.56 | 15.0% | 15.9% |
| ANN | 364.60 | 9.69 | 10.9% | 14.8% |
| LSTM | 355.74 | 9.82 | 13.1% | 13.6% |
| Hybrid | 357.50 | 9.77 | 12.6% | 14.1% |

![Per-component MAE by model](figures/mae_by_model_grouped_bar.png)

## Own-split comparison (each model on the split it was trained for)

Random-split floor (test): MSE=414.02, MAE=11.48. Chronological-split floor (test): MSE=409.24, MAE=11.37.

| model | split | test MSE | test MAE | % improvement over floor (MSE) | % improvement over floor (MAE) |
|---|---|---|---|---|---|
| XGBoost | random | 344.75 | 9.53 | 16.7% | 17.0% |
| ANN | random | 368.53 | 10.02 | 11.0% | 12.8% |
| LSTM | chronological | 355.74 | 9.82 | 13.1% | 13.6% |
| Hybrid | chronological | 357.50 | 9.77 | 12.6% | 14.1% |

### Per-component MAE (own split)

**XGBoost** (random split)

| component | MAE |
|---|---|
| DELAY_DUE_CARRIER | 16.33 |
| DELAY_DUE_WEATHER | 3.81 |
| DELAY_DUE_SECURITY | 0.30 |
| DELAY_DUE_NAS | 8.45 |
| DELAY_DUE_LATE_AIRCRAFT | 18.74 |

**ANN** (random split)

| component | MAE |
|---|---|
| DELAY_DUE_CARRIER | 16.94 |
| DELAY_DUE_WEATHER | 3.42 |
| DELAY_DUE_SECURITY | 0.33 |
| DELAY_DUE_NAS | 9.57 |
| DELAY_DUE_LATE_AIRCRAFT | 19.84 |

**LSTM** (chronological split)

| component | MAE |
|---|---|
| DELAY_DUE_CARRIER | 16.49 |
| DELAY_DUE_WEATHER | 3.61 |
| DELAY_DUE_SECURITY | 0.20 |
| DELAY_DUE_NAS | 9.60 |
| DELAY_DUE_LATE_AIRCRAFT | 19.22 |

**Hybrid** (chronological split)

| component | MAE |
|---|---|
| DELAY_DUE_CARRIER | 16.56 |
| DELAY_DUE_WEATHER | 3.42 |
| DELAY_DUE_SECURITY | 0.28 |
| DELAY_DUE_NAS | 9.28 |
| DELAY_DUE_LATE_AIRCRAFT | 19.32 |

## Prediction spread vs. truth (own split)

The source notebook noted that its LSTM's predictions had much lower variance than the actual delays. Checked here for all four models: predicted standard deviation vs. actual standard deviation, per component.

**XGBoost**

| component | truth mean | truth std | pred mean | pred std |
|---|---|---|---|---|
| DELAY_DUE_CARRIER | 15.99 | 26.11 | 16.14 | 8.87 |
| DELAY_DUE_WEATHER | 2.03 | 11.59 | 2.06 | 2.57 |
| DELAY_DUE_SECURITY | 0.13 | 2.30 | 0.13 | 0.67 |
| DELAY_DUE_NAS | 11.30 | 20.01 | 11.34 | 12.63 |
| DELAY_DUE_LATE_AIRCRAFT | 18.11 | 29.12 | 18.26 | 10.80 |

**ANN**

| component | truth mean | truth std | pred mean | pred std |
|---|---|---|---|---|
| DELAY_DUE_CARRIER | 15.99 | 26.11 | 15.30 | 5.88 |
| DELAY_DUE_WEATHER | 2.03 | 11.59 | 1.55 | 0.40 |
| DELAY_DUE_SECURITY | 0.13 | 2.30 | 0.17 | 0.14 |
| DELAY_DUE_NAS | 11.30 | 20.01 | 11.91 | 11.27 |
| DELAY_DUE_LATE_AIRCRAFT | 18.11 | 29.12 | 18.68 | 8.23 |

**LSTM**

| component | truth mean | truth std | pred mean | pred std |
|---|---|---|---|---|
| DELAY_DUE_CARRIER | 15.88 | 25.57 | 15.71 | 7.24 |
| DELAY_DUE_WEATHER | 1.98 | 11.41 | 1.87 | 0.97 |
| DELAY_DUE_SECURITY | 0.13 | 2.42 | 0.06 | 0.06 |
| DELAY_DUE_NAS | 10.91 | 19.66 | 12.54 | 11.43 |
| DELAY_DUE_LATE_AIRCRAFT | 19.37 | 29.45 | 18.03 | 9.13 |

**Hybrid**

| component | truth mean | truth std | pred mean | pred std |
|---|---|---|---|---|
| DELAY_DUE_CARRIER | 15.88 | 25.57 | 15.68 | 6.95 |
| DELAY_DUE_WEATHER | 1.98 | 11.41 | 1.63 | 1.02 |
| DELAY_DUE_SECURITY | 0.13 | 2.42 | 0.06 | 0.18 |
| DELAY_DUE_NAS | 10.91 | 19.66 | 11.74 | 11.05 |
| DELAY_DUE_LATE_AIRCRAFT | 19.37 | 29.45 | 18.04 | 9.29 |

**Reproduces**: predicted standard deviation is lower than actual standard deviation for every model and every component (confirmed). All four models predict a narrower range than the true delay distribution -- expected, given every feature correlates with ARR_DELAY below r=0.08: with this little signal, MSE-minimizing models converge toward predicting something close to the mean, under-representing the tails.

