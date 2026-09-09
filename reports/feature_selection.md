# Feature Selection Statistics

## 3a. Pearson correlation (non-categorical columns vs ARR_DELAY)

| column | r | p-value |
|---|---|---|
| CRS_DEP_TIME | 0.0704 | 0.000e+00 |
| TAXI_OUT | 0.0541 | 3.082e-314 |
| CRS_ARR_TIME | 0.0500 | 6.454e-269 |
| TAXI_IN | 0.0235 | 5.235e-61 |
| CRS_ELAPSED_TIME | -0.0122 | 1.304e-17 |
| DISTANCE | -0.0228 | 1.598e-57 |

CRS_ELAPSED_TIME vs DISTANCE (multicollinearity check): r = 0.9823.

## 3b. Kruskal-Wallis H test (categorical/identifier columns vs ARR_DELAY)

| column | H | p-value | n_groups |
|---|---|---|---|
| FL_DATE | 11695.48 | 0.000e+00 | 1704 |
| FL_NUMBER | 8809.14 | 1.016e-54 | 6831 |
| DOT_CODE | 4954.99 | 0.000e+00 | 18 |
| AIRLINE | 4954.99 | 0.000e+00 | 18 |
| AIRLINE_DOT | 4954.99 | 0.000e+00 | 18 |
| AIRLINE_CODE | 4954.99 | 0.000e+00 | 18 |
| ORIGIN | 4605.39 | 0.000e+00 | 380 |
| ORIGIN_CITY | 4345.32 | 0.000e+00 | 373 |
| DEST | 3988.84 | 0.000e+00 | 379 |
| DEST_CITY | 3744.04 | 0.000e+00 | 372 |

### Redundancy check for proposed drops

| col_a | col_b | n_unique(a) | n_unique(b) | n_pairs | one_to_one |
|---|---|---|---|---|---|
| AIRLINE | AIRLINE_DOT | 18 | 18 | 18 | True |
| AIRLINE | AIRLINE_CODE | 18 | 18 | 18 | True |
| AIRLINE | DOT_CODE | 18 | 18 | 18 | True |
| ORIGIN | ORIGIN_CITY | 380 | 373 | 381 | False |
| DEST | DEST_CITY | 379 | 372 | 380 | False |

## 3c. PCA check (six numeric columns, unscaled)

| n_components | per-component correlation (PC1, PC2, ...) |
|---|---|
| 1 | 0.0687 |
| 2 | 0.0687, -0.0008 |
| 3 | 0.0687, -0.0008, 0.0274 |
| 4 | 0.0687, -0.0008, 0.0274, 0.0757 |

## Decisions carried into feature engineering

- **Drop `CRS_ELAPSED_TIME`**: redundant with `DISTANCE` (r = 0.9823 between the two columns).
- **Drop `AIRLINE_DOT`**: one-to-one with `AIRLINE` (18 vs 18 unique values, 18 distinct pairs).
- **Drop `AIRLINE_CODE`**: one-to-one with `AIRLINE` (18 vs 18 unique values, 18 distinct pairs).
- **Drop `DOT_CODE`**: one-to-one with `AIRLINE` (18 vs 18 unique values, 18 distinct pairs).
- **Drop `ORIGIN_CITY`**: not strictly one-to-one, but near-redundant with `ORIGIN` (380 vs 373 unique values, 381 distinct pairs).
- **Drop `DEST_CITY`**: not strictly one-to-one, but near-redundant with `DEST` (379 vs 372 unique values, 380 distinct pairs).
- **Do not use PCA downstream**: components show no meaningful correlation improvement over the raw Pearson table above.
- **Keep** `CRS_DEP_TIME`, `TAXI_OUT`, `TAXI_IN`, `CRS_ARR_TIME`, `DISTANCE`, `FL_DATE`, `AIRLINE`, `FL_NUMBER`, `ORIGIN`, `DEST` — each carries a distinct, non-redundant Pearson/Kruskal-Wallis signal above.
