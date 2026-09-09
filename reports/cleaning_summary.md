# Cleaning Summary

1. Read CSV: 3000000 rows
2. Drop cancelled/diverted: 2913804 rows
3. Drop duplicates: 2913804 rows (0 duplicates removed)
4. Split by delay components: has_components=533863, missing_components=2379941
5. Keep has_components: 533863 rows (retained fraction 0.1832)
6. Integrity check (components sum to ARR_DELAY): 0 violations, 533863 rows
7. IQR prune on ARR_DELAY: 489828 rows

## ARR_DELAY before/after IQR pruning

| stat | before | after |
|---|---|---|
| count | 533863.00 | 489828.00 |
| mean | 67.53 | 47.83 |
| std | 93.91 | 32.87 |
| min | 15.00 | 15.00 |
| max | 2934.00 | 154.00 |

Removed 44035 rows (8.25%)

8. Saved 489828 rows to /Users/akshatgarg/Desktop/flight-delay-prediction/data/processed/flights_clean.parquet
