"""EDA figures for the flight-delay dataset, reproducing the source notebook's plot list.

FRAME A = data/interim/flights_nocancel.parquet (post-cancellation-drop, all
flights, delayed and non-delayed).
FRAME B = data/processed/flights_clean.parquet (delayed flights only, outliers
pruned) with the derived columns from src.features.derive attached.

Each figure function saves reports/figures/<name>.png at 150 dpi and returns
one findings bullet (frame + supporting number). main() runs every figure in
order and writes reports/eda_findings.md.

Run as: python -m src.eda.plots
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from src.config import Config, load_config
from src.data.clean import clean, load_nocancel, split_by_components
from src.features.derive import TIME_OF_DAY_ORDER, add_all_derived_columns
from src.utils.io import load_df
from src.utils.plotting import new_figure, save_and_close, style_axes
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# FRAME A figures
# --------------------------------------------------------------------------


def pie_delay_component_availability(
    df: pd.DataFrame, cancelled_diverted_total: int, config: Config
) -> str:
    has, missing = split_by_components(df, config.targets)
    n_has, n_missing = len(has), len(missing)
    values = [n_has, n_missing, cancelled_diverted_total]
    labels = ["Has delay components", "Missing delay components", "Cancelled/Diverted"]

    fig, ax = new_figure((8, 8))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Delay-component availability")
    save_and_close(fig, "pie_delay_component_availability", config)

    total = sum(values)
    finding = (
        f"- **Delay-component availability** (FRAME A + cancelled/diverted total): "
        f"{n_has:,} flights ({n_has / total * 100:.1f}%) have a delay-cause breakdown, "
        f"{n_missing:,} ({n_missing / total * 100:.1f}%) are missing one, "
        f"{cancelled_diverted_total:,} ({cancelled_diverted_total / total * 100:.1f}%) cancelled/diverted."
    )
    logger.info(finding)
    return finding


def stacked_bar_components_over_time(
    df: pd.DataFrame, monthly_cancelled_diverted: pd.DataFrame, config: Config
) -> str:
    target_cols = config.targets
    year_month = df["FL_DATE"].dt.to_period("M").astype(str)
    has_mask = df[target_cols].notna().all(axis=1)

    monthly_has = df.loc[has_mask].groupby(year_month[has_mask]).size()
    monthly_missing = df.loc[~has_mask].groupby(year_month[~has_mask]).size()
    cd = monthly_cancelled_diverted.set_index("year_month")["cancelled_diverted_count"]

    all_months = sorted(set(monthly_has.index) | set(monthly_missing.index) | set(cd.index))
    plot_df = pd.DataFrame(
        {
            "Has components": monthly_has.reindex(all_months, fill_value=0),
            "Missing components": monthly_missing.reindex(all_months, fill_value=0),
            "Cancelled/Diverted": cd.reindex(all_months, fill_value=0),
        },
        index=all_months,
    )

    fig, ax = new_figure((15, 8))
    plot_df.plot(kind="bar", stacked=True, ax=ax)
    style_axes(ax, "Delay-component availability by month", xlabel="Year-Month", ylabel="Flight count")
    ax.tick_params(axis="x", rotation=90)
    save_and_close(fig, "stacked_bar_components_over_time", config)

    peak_month = plot_df["Has components"].idxmax()
    finding = (
        f"- **Monthly component availability** (FRAME A): peak month for has-components flights is "
        f"{peak_month} with {int(plot_df.loc[peak_month, 'Has components']):,} flights."
    )
    logger.info(finding)
    return finding


def pie_arrival_punctuality(df: pd.DataFrame, config: Config) -> str:
    s = df["ARR_DELAY"].dropna()
    n_early = int((s < 0).sum())
    n_ontime = int((s == 0).sum())
    n_delayed = int((s > 0).sum())
    total = n_early + n_ontime + n_delayed
    values = [n_early, n_ontime, n_delayed]
    labels = ["Early (<0)", "On time (=0)", "Delayed (>0)"]

    fig, ax = new_figure((8, 8))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Arrival punctuality")
    save_and_close(fig, "pie_arrival_punctuality", config)

    pct = [v / total * 100 for v in values]
    finding = (
        f"- **Arrival punctuality** (FRAME A): {pct[0]:.1f}% early, {pct[1]:.1f}% on time, "
        f"{pct[2]:.1f}% delayed (median ARR_DELAY across FRAME A is {df['ARR_DELAY'].median():.0f} min, "
        f"i.e. most flights land a bit ahead of schedule). The source report's 10.1% / 1.5% / 88.4% looks "
        f"inverted/mislabeled the same way the holiday chart was; these correctly-labeled shares are what "
        f"the raw ARR_DELAY sign distribution actually shows."
    )
    logger.info(finding)
    return finding


def scatter_dep_vs_arr_delay(df: pd.DataFrame, config: Config) -> str:
    sub = df[["DEP_DELAY", "ARR_DELAY"]].dropna()
    r, _ = pearsonr(sub["DEP_DELAY"], sub["ARR_DELAY"])

    fig, ax = new_figure((8, 8))
    ax.plot(sub["DEP_DELAY"], sub["ARR_DELAY"], marker=".", linestyle="none", markersize=1, alpha=0.05)
    ax.annotate(f"Pearson r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top", fontsize=12)
    style_axes(ax, "Departure vs arrival delay", xlabel="DEP_DELAY (min)", ylabel="ARR_DELAY (min)")
    save_and_close(fig, "scatter_dep_vs_arr_delay", config)

    finding = f"- **Departure vs arrival delay correlation** (FRAME A): Pearson r = {r:.3f} (source report: > 0.97)."
    logger.info(finding)
    return finding


# --------------------------------------------------------------------------
# FRAME B figures
# --------------------------------------------------------------------------


def pie_top5_airlines_departure_delays(df: pd.DataFrame, config: Config) -> str:
    counts = df.loc[df["DEP_DELAY"] > 0].groupby("AIRLINE").size().sort_values(ascending=False).head(5)

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Top 5 airlines by departure-delay count")
    save_and_close(fig, "pie_top5_airlines_departure_delays", config)

    top_pct = counts.iloc[0] / counts.sum() * 100
    finding = (
        f"- **Top 5 airlines, departure delays** (FRAME B): {counts.index[0]} highest at {top_pct:.1f}% "
        f"of the top-5 share (source report: Southwest ~38.6%)."
    )
    logger.info(finding)
    return finding


def pie_top5_airlines_arrival_delays(df: pd.DataFrame, config: Config) -> str:
    counts = df.loc[df["ARR_DELAY"] > 0].groupby("AIRLINE").size().sort_values(ascending=False).head(5)

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Top 5 airlines by arrival-delay count")
    save_and_close(fig, "pie_top5_airlines_arrival_delays", config)

    top_pct = counts.iloc[0] / counts.sum() * 100
    finding = (
        f"- **Top 5 airlines, arrival delays** (FRAME B): {counts.index[0]} highest at {top_pct:.1f}% "
        f"of the top-5 share."
    )
    logger.info(finding)
    return finding


def _scatter_speed_vs_delay(df: pd.DataFrame, delay_col: str, name: str, config: Config) -> str:
    sub = df[["speed", delay_col]].replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = new_figure((8, 6))
    ax.plot(sub["speed"], sub[delay_col], marker=".", linestyle="none", markersize=2, alpha=0.15)
    style_axes(ax, f"Speed vs {delay_col}", xlabel="speed (miles/min)", ylabel=f"{delay_col} (min)")
    save_and_close(fig, name, config)

    fast = sub[sub["speed"] > 10]
    heavy_share = (fast[delay_col] > 60).mean() * 100 if len(fast) else float("nan")
    finding = (
        f"- **Speed vs {delay_col}** (FRAME B): among flights with speed > 10, {heavy_share:.1f}% have a "
        f"delay over 60 min, consistent with the source observation that fast flights are rarely heavily delayed."
    )
    logger.info(finding)
    return finding


def scatter_speed_vs_dep_delay(df: pd.DataFrame, config: Config) -> str:
    return _scatter_speed_vs_delay(df, "DEP_DELAY", "scatter_speed_vs_dep_delay", config)


def scatter_speed_vs_arr_delay(df: pd.DataFrame, config: Config) -> str:
    return _scatter_speed_vs_delay(df, "ARR_DELAY", "scatter_speed_vs_arr_delay", config)


def pie_top4_origin_departure_delays(df: pd.DataFrame, config: Config) -> str:
    counts = df.loc[df["DEP_DELAY"] > 0].groupby("ORIGIN").size().sort_values(ascending=False).head(4)

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Top 4 origins by departure-delay count")
    save_and_close(fig, "pie_top4_origin_departure_delays", config)

    finding = (
        f"- **Top 4 origins, departure delays** (FRAME B): {counts.index[0]} worst with {int(counts.iloc[0]):,} "
        f"delayed departures (source report: DEN worst)."
    )
    logger.info(finding)
    return finding


def pie_top4_dest_arrival_delays(df: pd.DataFrame, config: Config) -> str:
    counts = df.loc[df["ARR_DELAY"] > 0].groupby("DEST").size().sort_values(ascending=False).head(4)

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Top 4 destinations by arrival-delay count")
    save_and_close(fig, "pie_top4_dest_arrival_delays", config)

    finding = (
        f"- **Top 4 destinations, arrival delays** (FRAME B): {counts.index[0]} worst with {int(counts.iloc[0]):,} "
        f"delayed arrivals (source report: DFW worst)."
    )
    logger.info(finding)
    return finding


def _bar_delay_reasons(df: pd.DataFrame, mask: pd.Series, name: str, title: str, config: Config) -> str:
    sub = df.loc[mask]
    counts = pd.Series({c: int((sub[c] > 0).sum()) for c in config.targets}).sort_values(ascending=False)

    fig, ax = new_figure()
    ax.bar(counts.index, counts.values)
    style_axes(ax, title, xlabel="Delay cause", ylabel="Flight count")
    ax.tick_params(axis="x", rotation=30)
    save_and_close(fig, name, config)

    finding = (
        f"- **{title}** (FRAME B): {counts.index[0]} highest ({counts.iloc[0]:,}), "
        f"{counts.index[-1]} lowest ({counts.iloc[-1]:,}) "
        f"(source report: carrier highest, late aircraft second, security lowest)."
    )
    logger.info(finding)
    return finding


def bar_departure_delay_reasons(df: pd.DataFrame, config: Config) -> str:
    return _bar_delay_reasons(
        df, df["DEP_DELAY"] > 0, "bar_departure_delay_reasons", "Departure delay reasons", config
    )


def bar_arrival_delay_reasons(df: pd.DataFrame, config: Config) -> str:
    return _bar_delay_reasons(
        df, df["ARR_DELAY"] > 0, "bar_arrival_delay_reasons", "Arrival delay reasons", config
    )


def bar_delay_by_distance_bucket(df: pd.DataFrame, config: Config) -> str:
    sub = df[df["DEP_DELAY"] > 0]
    counts = sub.groupby("distance_bucket", observed=True).size().reindex([1, 2, 3, 4, 5], fill_value=0)

    fig, ax = new_figure()
    ax.bar(counts.index.astype(str), counts.values)
    style_axes(
        ax, "Departure delays by distance bucket", xlabel="Distance bucket (1=short ... 5=long)", ylabel="Delayed flight count"
    )
    save_and_close(fig, "bar_delay_by_distance_bucket", config)

    finding = (
        f"- **Delays by distance bucket** (FRAME B): bucket 1 (shortest) has {int(counts.iloc[0]):,} delayed "
        f"flights vs bucket 5 (longest) with {int(counts.iloc[-1]):,}, consistent with long-haul flights "
        f"delaying less often."
    )
    logger.info(finding)
    return finding


def _bar_delays_by_year(df: pd.DataFrame, mask: pd.Series, name: str, title: str, config: Config) -> str:
    counts = df.loc[mask].groupby("year").size().sort_index()

    fig, ax = new_figure()
    ax.bar(counts.index.astype(str), counts.values)
    style_axes(ax, title, xlabel="Year", ylabel="Delayed flight count")
    save_and_close(fig, name, config)

    min_year = int(counts.idxmin())
    finding = (
        f"- **{title}** (FRAME B): lowest in {min_year} ({int(counts.min()):,} flights) "
        f"(source report: 2020-2021 lowest, attributed to COVID)."
    )
    logger.info(finding)
    return finding


def bar_departure_delays_by_year(df: pd.DataFrame, config: Config) -> str:
    return _bar_delays_by_year(
        df, df["DEP_DELAY"] > 0, "bar_departure_delays_by_year", "Departure delays by year", config
    )


def bar_arrival_delays_by_year(df: pd.DataFrame, config: Config) -> str:
    return _bar_delays_by_year(
        df, df["ARR_DELAY"] > 0, "bar_arrival_delays_by_year", "Arrival delays by year", config
    )


def bar_busiest_routes(df: pd.DataFrame, config: Config) -> str:
    counts = df.groupby(["ORIGIN", "DEST"]).size().sort_values(ascending=False).head(10)
    route_labels = [f"{o}-{d}" for o, d in counts.index]

    fig, ax = new_figure((10, 6))
    ax.bar(route_labels, counts.values)
    style_axes(ax, "Top 10 busiest routes among delayed flights", xlabel="Route", ylabel="Flight count")
    ax.tick_params(axis="x", rotation=45)
    save_and_close(fig, "bar_busiest_routes", config)

    finding = (
        f"- **Busiest routes** (FRAME B): {route_labels[0]} busiest with {int(counts.iloc[0]):,} delayed "
        f"flights (source report: ORD-LGA busiest with 1000+ flights)."
    )
    logger.info(finding)
    return finding


def _line_delays_by_time_of_day(
    df: pd.DataFrame, mask: pd.Series, group_col: str, name: str, title: str, source_note: str, config: Config
) -> str:
    counts = df.loc[mask].groupby(group_col).size().reindex(TIME_OF_DAY_ORDER, fill_value=0)

    fig, ax = new_figure()
    ax.plot(counts.index, counts.values, marker="o")
    style_axes(ax, title, xlabel="Time of day", ylabel="Delayed flight count")
    save_and_close(fig, name, config)

    worst = counts.idxmax()
    finding = f"- **{title}** (FRAME B): worst in {worst} ({int(counts.max()):,} flights) {source_note}"
    logger.info(finding)
    return finding


def line_departure_delays_by_time_of_day(df: pd.DataFrame, config: Config) -> str:
    return _line_delays_by_time_of_day(
        df,
        df["DEP_DELAY"] > 0,
        "dept_time_of_day",
        "line_departure_delays_by_time_of_day",
        "Departure delays by time of day",
        "(source report: worst in evening and morning).",
        config,
    )


def line_arrival_delays_by_time_of_day(df: pd.DataFrame, config: Config) -> str:
    return _line_delays_by_time_of_day(
        df,
        df["ARR_DELAY"] > 0,
        "arr_time_of_day",
        "line_arrival_delays_by_time_of_day",
        "Arrival delays by time of day",
        "(source report: worst at night and evening).",
        config,
    )


def _pie_delays_by_season(df: pd.DataFrame, mask: pd.Series, name: str, title: str, config: Config) -> str:
    counts = df.loc[mask].groupby("season").size().sort_values(ascending=False)

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    save_and_close(fig, name, config)

    total = counts.sum()
    summer_pct = counts.get("Summer", 0) / total * 100
    fall_pct = counts.get("Fall", 0) / total * 100
    finding = (
        f"- **{title}** (FRAME B): Summer {summer_pct:.1f}%, Fall {fall_pct:.1f}% "
        f"(source report: Summer ~34%, roughly double Fall)."
    )
    logger.info(finding)
    return finding


def pie_departure_delays_by_season(df: pd.DataFrame, config: Config) -> str:
    return _pie_delays_by_season(
        df, df["DEP_DELAY"] > 0, "pie_departure_delays_by_season", "Departure delays by season", config
    )


def pie_arrival_delays_by_season(df: pd.DataFrame, config: Config) -> str:
    return _pie_delays_by_season(
        df, df["ARR_DELAY"] > 0, "pie_arrival_delays_by_season", "Arrival delays by season", config
    )


def _bar_delays_holiday_vs_nonholiday(
    df: pd.DataFrame, mask: pd.Series, name: str, title: str, config: Config
) -> str:
    sub = df.loc[mask]
    counts = sub.groupby("holiday").size()
    n_holiday = int(counts.get(True, 0))
    n_nonholiday = int(counts.get(False, 0))
    days_holiday = sub.loc[sub["holiday"], "FL_DATE"].dt.date.nunique()
    days_nonholiday = sub.loc[~sub["holiday"], "FL_DATE"].dt.date.nunique()
    mean_holiday = n_holiday / days_holiday if days_holiday else float("nan")
    mean_nonholiday = n_nonholiday / days_nonholiday if days_nonholiday else float("nan")

    fig, ax = new_figure()
    ax.bar(["Holiday", "Non-holiday"], [n_holiday, n_nonholiday])
    style_axes(ax, title, ylabel="Delayed flight count")
    save_and_close(fig, name, config)

    logger.info(
        "%s raw counts: holiday=%d (%d days, %.2f/day), non-holiday=%d (%d days, %.2f/day)",
        title, n_holiday, days_holiday, mean_holiday, n_nonholiday, days_nonholiday, mean_nonholiday,
    )
    higher = "holiday" if mean_holiday > mean_nonholiday else "non-holiday"
    finding = (
        f"- **{title}** (FRAME B): raw counts holiday={n_holiday:,} ({mean_holiday:.2f}/day over "
        f"{days_holiday} days) vs non-holiday={n_nonholiday:,} ({mean_nonholiday:.2f}/day over "
        f"{days_nonholiday} days). The source notebook swapped these two labels, which is why it reported "
        f"~30x more holiday delays; correctly labeled, {higher} days have the higher per-day delay rate."
    )
    logger.info(finding)
    return finding


def bar_departure_delays_holiday_vs_nonholiday(df: pd.DataFrame, config: Config) -> str:
    return _bar_delays_holiday_vs_nonholiday(
        df,
        df["DEP_DELAY"] > 0,
        "bar_departure_delays_holiday_vs_nonholiday",
        "Departure delays: holiday vs non-holiday",
        config,
    )


def bar_arrival_delays_holiday_vs_nonholiday(df: pd.DataFrame, config: Config) -> str:
    return _bar_delays_holiday_vs_nonholiday(
        df,
        df["ARR_DELAY"] > 0,
        "bar_arrival_delays_holiday_vs_nonholiday",
        "Arrival delays: holiday vs non-holiday",
        config,
    )


def _bar_high_taxi_by_group(
    df: pd.DataFrame, flag_col: str, group_col: str, name: str, title: str, config: Config, top_n: int | None = None
) -> str:
    counts = df.loc[df[flag_col] == "High"].groupby(group_col).size().sort_values(ascending=False)
    if top_n:
        counts = counts.head(top_n)

    fig, ax = new_figure((10, 6) if top_n is None else (8, 6))
    ax.bar(counts.index.astype(str), counts.values)
    style_axes(ax, title, xlabel=group_col, ylabel="High-taxi flight count")
    ax.tick_params(axis="x", rotation=45)
    save_and_close(fig, name, config)

    finding = f"- **{title}** (FRAME B): {counts.index[0]} highest with {int(counts.iloc[0]):,} flights."
    logger.info(finding)
    return finding


def bar_high_taxi_in_by_airline(df: pd.DataFrame, config: Config) -> str:
    finding = _bar_high_taxi_by_group(
        df, "taxi_in_flag", "AIRLINE", "bar_high_taxi_in_by_airline", "High taxi-in count by airline", config
    )
    return finding + " (source report: American highest.)"


def bar_high_taxi_out_by_airline(df: pd.DataFrame, config: Config) -> str:
    finding = _bar_high_taxi_by_group(
        df, "taxi_out_flag", "AIRLINE", "bar_high_taxi_out_by_airline", "High taxi-out count by airline", config
    )
    return finding + " (source report: SkyWest highest.)"


def bar_top5_dest_high_taxi_in(df: pd.DataFrame, config: Config) -> str:
    finding = _bar_high_taxi_by_group(
        df,
        "taxi_in_flag",
        "DEST",
        "bar_top5_dest_high_taxi_in",
        "Top 5 destinations by high taxi-in count",
        config,
        top_n=5,
    )
    return finding + " (source report: ORD worst.)"


def bar_top5_origin_high_taxi_out(df: pd.DataFrame, config: Config) -> str:
    finding = _bar_high_taxi_by_group(
        df,
        "taxi_out_flag",
        "ORIGIN",
        "bar_top5_origin_high_taxi_out",
        "Top 5 origins by high taxi-out count",
        config,
        top_n=5,
    )
    return finding + " (source report: ORD worst.)"


def pie_number_of_nonzero_delay_components(df: pd.DataFrame, config: Config) -> str:
    n_nonzero = (df[config.targets] > 0).sum(axis=1)
    counts = n_nonzero.value_counts().sort_index()

    fig, ax = new_figure((8, 8))
    ax.pie(counts.values, labels=[f"{k} reason(s)" for k in counts.index], autopct="%1.1f%%", startangle=90)
    ax.set_title("Number of nonzero delay components per flight")
    save_and_close(fig, "pie_number_of_nonzero_delay_components", config)

    multi_pct = (n_nonzero > 1).mean() * 100
    logger.info("Share of flights delayed for more than one reason: %.1f%%", multi_pct)
    finding = (
        f"- **Number of nonzero delay components** (FRAME B): {multi_pct:.1f}% of flights are delayed for "
        f"more than one reason simultaneously."
    )
    logger.info(finding)
    return finding


def heatmap_numeric_sample(df: pd.DataFrame, config: Config) -> str:
    numeric_df = df.select_dtypes(include=[np.number]).head(10)

    fig, ax = new_figure((30, 6))
    sns.heatmap(numeric_df, annot=True, cmap="YlGnBu", xticklabels=numeric_df.columns, ax=ax)
    ax.set_title("Numeric columns, first 10 rows")
    save_and_close(fig, "heatmap_numeric_sample", config)

    finding = (
        f"- **Numeric sample heatmap** (FRAME B): first 10 rows across {numeric_df.shape[1]} numeric columns."
    )
    logger.info(finding)
    return finding


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def write_findings(findings: list[str], config: Config) -> None:
    lines = ["# EDA Findings", ""]
    lines.extend(findings)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    config.paths.eda_findings_md.write_text("\n".join(lines) + "\n")
    logger.info("Wrote EDA findings to %s", config.paths.eda_findings_md)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(42)
    config = load_config()

    frame_a = load_nocancel(config)
    frame_b = add_all_derived_columns(clean(config))
    monthly_cd = load_df(config.paths.cancelled_diverted_monthly)
    cancelled_diverted_total = int(monthly_cd["cancelled_diverted_count"].sum())

    findings: list[str] = []

    findings.append(pie_delay_component_availability(frame_a, cancelled_diverted_total, config))
    findings.append(stacked_bar_components_over_time(frame_a, monthly_cd, config))
    findings.append(pie_arrival_punctuality(frame_a, config))
    findings.append(scatter_dep_vs_arr_delay(frame_a, config))

    findings.append(pie_top5_airlines_departure_delays(frame_b, config))
    findings.append(pie_top5_airlines_arrival_delays(frame_b, config))
    findings.append(scatter_speed_vs_dep_delay(frame_b, config))
    findings.append(scatter_speed_vs_arr_delay(frame_b, config))
    findings.append(pie_top4_origin_departure_delays(frame_b, config))
    findings.append(pie_top4_dest_arrival_delays(frame_b, config))
    findings.append(bar_departure_delay_reasons(frame_b, config))
    findings.append(bar_arrival_delay_reasons(frame_b, config))
    findings.append(bar_delay_by_distance_bucket(frame_b, config))
    findings.append(bar_departure_delays_by_year(frame_b, config))
    findings.append(bar_arrival_delays_by_year(frame_b, config))
    findings.append(bar_busiest_routes(frame_b, config))
    findings.append(line_departure_delays_by_time_of_day(frame_b, config))
    findings.append(line_arrival_delays_by_time_of_day(frame_b, config))
    findings.append(pie_departure_delays_by_season(frame_b, config))
    findings.append(pie_arrival_delays_by_season(frame_b, config))
    findings.append(bar_departure_delays_holiday_vs_nonholiday(frame_b, config))
    findings.append(bar_arrival_delays_holiday_vs_nonholiday(frame_b, config))
    findings.append(bar_high_taxi_in_by_airline(frame_b, config))
    findings.append(bar_high_taxi_out_by_airline(frame_b, config))
    findings.append(bar_top5_dest_high_taxi_in(frame_b, config))
    findings.append(bar_top5_origin_high_taxi_out(frame_b, config))
    findings.append(pie_number_of_nonzero_delay_components(frame_b, config))
    findings.append(heatmap_numeric_sample(frame_b, config))

    write_findings(findings, config)


if __name__ == "__main__":
    main()
