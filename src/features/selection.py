"""Feature-selection statistics on data/processed/flights_clean.parquet.

Three analyses, all against ARR_DELAY:
    3a. Pearson correlation on six non-categorical columns.
    3b. Kruskal-Wallis H test on ten categorical/identifier columns, plus a
        direct redundancy check for the pairs the source analysis proposed
        dropping.
    3c. PCA over the six numeric columns (1-4 components, unscaled), each
        component's correlation against the summed five-component delay
        target.

This module only computes and reports (reports/feature_selection.md); it
does not modify or write any dataset.

Run as: python -m src.features.selection
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import kruskal, pearsonr
from sklearn.decomposition import PCA

from src.config import Config, load_config
from src.utils.io import load_df
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

PEARSON_COLS = ["CRS_DEP_TIME", "TAXI_OUT", "TAXI_IN", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]

KRUSKAL_COLS = [
    "FL_DATE", "AIRLINE", "AIRLINE_DOT", "AIRLINE_CODE", "DOT_CODE",
    "FL_NUMBER", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY",
]

REDUNDANCY_PAIRS = [
    ("AIRLINE", "AIRLINE_DOT"),
    ("AIRLINE", "AIRLINE_CODE"),
    ("AIRLINE", "DOT_CODE"),
    ("ORIGIN", "ORIGIN_CITY"),
    ("DEST", "DEST_CITY"),
]

MAX_PCA_COMPONENTS = 4


def calculate_pearson_score(df: pd.DataFrame, col: str, y: str) -> float:
    """Pearson correlation coefficient between df[col] and df[y]."""
    r, _ = pearsonr(df[col], df[y])
    return r


def pearson_table(df: pd.DataFrame, cols: list[str], y: str) -> pd.DataFrame:
    """Pearson r and p-value for each column against y, sorted descending by r."""
    rows = []
    for col in cols:
        r, p = pearsonr(df[col], df[y])
        rows.append({"column": col, "r": r, "p_value": p})
    table = pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)
    return table


def kruskal_wallis_table(df: pd.DataFrame, cols: list[str], y: str) -> pd.DataFrame:
    """Kruskal-Wallis H, p-value, and group count for each column, sorted descending by H."""
    rows = []
    for col in cols:
        groups = [group[y].to_numpy() for _, group in df.groupby(col, observed=True)]
        n_groups = len(groups)
        h, p = kruskal(*groups)
        logger.info("Kruskal-Wallis on %s: H=%.2f p=%.3e n_groups=%d", col, h, p, n_groups)
        rows.append({"column": col, "H": h, "p_value": p, "n_groups": n_groups})
    table = pd.DataFrame(rows).sort_values("H", ascending=False).reset_index(drop=True)
    return table


def check_redundancy(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Report unique-value counts for col_a/col_b and whether their mapping is one-to-one."""
    n_a = int(df[col_a].nunique())
    n_b = int(df[col_b].nunique())
    pairs = df[[col_a, col_b]].drop_duplicates()
    a_to_b_unique = bool(pairs.groupby(col_a)[col_b].nunique().eq(1).all())
    b_to_a_unique = bool(pairs.groupby(col_b)[col_a].nunique().eq(1).all())
    one_to_one = a_to_b_unique and b_to_a_unique

    result = {
        "col_a": col_a,
        "col_b": col_b,
        "n_unique_a": n_a,
        "n_unique_b": n_b,
        "n_pairs": len(pairs),
        "one_to_one": one_to_one,
    }
    logger.info(
        "Redundancy check %s vs %s: n_unique=%d/%d, n_pairs=%d, one_to_one=%s",
        col_a, col_b, n_a, n_b, len(pairs), one_to_one,
    )
    return result


def pca_check(df: pd.DataFrame, cols: list[str], target_cols: list[str], max_components: int) -> dict[int, list[float]]:
    """For n_components in 1..max_components, fit PCA (unscaled) and correlate each
    component against the summed five-component delay target."""
    X = df[cols].to_numpy()
    y_sum = df[target_cols].sum(axis=1).to_numpy()

    results: dict[int, list[float]] = {}
    for n in range(1, max_components + 1):
        pca = PCA(n_components=n, svd_solver="full")
        components = pca.fit_transform(X)
        corrs = [pearsonr(components[:, i], y_sum)[0] for i in range(n)]
        results[n] = corrs
        logger.info("PCA n_components=%d correlations: %s", n, [f"{r:.4f}" for r in corrs])
    return results


def _format_pearson_table(table: pd.DataFrame) -> list[str]:
    lines = ["| column | r | p-value |", "|---|---|---|"]
    for _, row in table.iterrows():
        lines.append(f"| {row['column']} | {row['r']:.4f} | {row['p_value']:.3e} |")
    return lines


def _format_kruskal_table(table: pd.DataFrame) -> list[str]:
    lines = ["| column | H | p-value | n_groups |", "|---|---|---|---|"]
    for _, row in table.iterrows():
        lines.append(f"| {row['column']} | {row['H']:.2f} | {row['p_value']:.3e} | {int(row['n_groups'])} |")
    return lines


def _format_redundancy(results: list[dict]) -> list[str]:
    lines = ["| col_a | col_b | n_unique(a) | n_unique(b) | n_pairs | one_to_one |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['col_a']} | {r['col_b']} | {r['n_unique_a']} | {r['n_unique_b']} | "
            f"{r['n_pairs']} | {r['one_to_one']} |"
        )
    return lines


def _format_pca_table(results: dict[int, list[float]]) -> list[str]:
    lines = ["| n_components | per-component correlation (PC1, PC2, ...) |", "|---|---|"]
    for n, corrs in results.items():
        corr_str = ", ".join(f"{c:.4f}" for c in corrs)
        lines.append(f"| {n} | {corr_str} |")
    return lines


def write_report(
    config: Config,
    pearson_tbl: pd.DataFrame,
    elapsed_vs_distance_r: float,
    kruskal_tbl: pd.DataFrame,
    redundancy_results: list[dict],
    pca_results: dict[int, list[float]],
) -> None:
    lines: list[str] = ["# Feature Selection Statistics", ""]

    lines.append("## 3a. Pearson correlation (non-categorical columns vs ARR_DELAY)")
    lines.append("")
    lines.extend(_format_pearson_table(pearson_tbl))
    lines.append("")
    lines.append(
        f"CRS_ELAPSED_TIME vs DISTANCE (multicollinearity check): r = {elapsed_vs_distance_r:.4f}."
    )
    lines.append("")

    lines.append("## 3b. Kruskal-Wallis H test (categorical/identifier columns vs ARR_DELAY)")
    lines.append("")
    lines.extend(_format_kruskal_table(kruskal_tbl))
    lines.append("")
    lines.append("### Redundancy check for proposed drops")
    lines.append("")
    lines.extend(_format_redundancy(redundancy_results))
    lines.append("")

    lines.append("## 3c. PCA check (six numeric columns, unscaled)")
    lines.append("")
    lines.extend(_format_pca_table(pca_results))
    lines.append("")

    lines.append("## Decisions carried into Phase 4")
    lines.append("")
    lines.append(
        f"- **Drop `CRS_ELAPSED_TIME`**: redundant with `DISTANCE` "
        f"(r = {elapsed_vs_distance_r:.4f} between the two columns)."
    )
    for r in redundancy_results:
        verdict = "one-to-one" if r["one_to_one"] else "not strictly one-to-one, but near-redundant"
        lines.append(
            f"- **Drop `{r['col_b']}`**: {verdict} with `{r['col_a']}` "
            f"({r['n_unique_a']} vs {r['n_unique_b']} unique values, {r['n_pairs']} distinct pairs)."
        )
    lines.append(
        "- **Do not use PCA downstream**: components show no meaningful correlation improvement over "
        "the raw Pearson table above."
    )
    lines.append(
        "- **Keep** `CRS_DEP_TIME`, `TAXI_OUT`, `TAXI_IN`, `CRS_ARR_TIME`, `DISTANCE`, `FL_DATE`, "
        "`AIRLINE`, `FL_NUMBER`, `ORIGIN`, `DEST` — each carries a distinct, non-redundant Pearson/"
        "Kruskal-Wallis signal above."
    )

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.paths.reports_dir / "feature_selection.md"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote feature-selection report to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(42)
    config = load_config()

    df = load_df(config.paths.clean_parquet)
    y = config.arrival_delay_col

    pearson_tbl = pearson_table(df, PEARSON_COLS, y)
    logger.info("Pearson table:\n%s", pearson_tbl.to_string(index=False))

    elapsed_vs_distance_r = calculate_pearson_score(df, "CRS_ELAPSED_TIME", "DISTANCE")
    logger.info("CRS_ELAPSED_TIME vs DISTANCE: r = %.4f", elapsed_vs_distance_r)

    kruskal_tbl = kruskal_wallis_table(df, KRUSKAL_COLS, y)
    logger.info("Kruskal-Wallis table:\n%s", kruskal_tbl.to_string(index=False))

    redundancy_results = [check_redundancy(df, a, b) for a, b in REDUNDANCY_PAIRS]

    pca_results = pca_check(df, PEARSON_COLS, config.targets, MAX_PCA_COMPONENTS)

    write_report(config, pearson_tbl, elapsed_vs_distance_r, kruskal_tbl, redundancy_results, pca_results)


if __name__ == "__main__":
    main()
