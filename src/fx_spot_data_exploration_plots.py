"""Modular plotnine charts for FX spot exploration.

This module refactors the plotting logic from
`notebooks/fx_spot_data_exploration.ipynb` into reusable functions.
All charts are built with plotnine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_rect,
    geom_ribbon,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient2,
    scale_fill_manual,
    scale_x_datetime,
    theme,
    theme_minimal,
)

from src.fx_data import load_fx_spot

CURRENCIES = ["AUD", "BRL", "CAD", "GBP", "JPY", "MXN", "ZAR"]
CURRENCY_NAMES = {
    "AUD": "Australian Dollar",
    "BRL": "Brazilian Real",
    "CAD": "Canadian Dollar",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "MXN": "Mexican Peso",
    "ZAR": "South African Rand",
}

PALETTE = {
    "AUD": "#2563EB",
    "BRL": "#16A34A",
    "CAD": "#D97706",
    "GBP": "#7C3AED",
    "JPY": "#DC2626",
    "MXN": "#EA580C",
    "ZAR": "#0891B2",
}

SHADING = [
    ("2008-09-01", "2009-06-30", "GFC"),
    ("2020-02-15", "2020-06-30", "COVID-19"),
]


def prepare_fx_plot_data(
    df: pl.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (long, wide, log_returns) pandas DataFrames for plotting."""
    long_df = df.to_pandas().copy()
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["date", "currency"])

    wide_df = (
        long_df.pivot(index="date", columns="currency", values="rate_per_usd")
        .sort_index()
        .reindex(columns=CURRENCIES)
    )
    log_ret = np.log(wide_df / wide_df.shift(1))
    return long_df, wide_df, log_ret


def _shading_df() -> pd.DataFrame:
    return pd.DataFrame(SHADING, columns=["start", "end", "label"]).assign(
        start=lambda d: pd.to_datetime(d["start"]),
        end=lambda d: pd.to_datetime(d["end"]),
    )


def plot_indexed_spot_rates(wide_df: pd.DataFrame) -> ggplot:
    """Figure 1: Indexed spot rates (Jan 1995 = 100)."""
    subset = wide_df.loc["1995-01-01":, CURRENCIES].copy()
    base = subset.bfill().iloc[0]
    indexed = subset.div(base).mul(100)

    plot_df = (
        indexed.reset_index()
        .melt(id_vars="date", var_name="currency", value_name="index_level")
        .dropna()
    )
    plot_df["currency"] = pd.Categorical(plot_df["currency"], CURRENCIES, ordered=True)

    return (
        ggplot(plot_df, aes("date", "index_level", color="currency"))
        + geom_rect(
            data=_shading_df(),
            mapping=aes(xmin="start", xmax="end", ymin=-np.inf, ymax=np.inf),
            inherit_aes=False,
            fill="#374151",
            alpha=0.07,
        )
        + geom_line(size=0.8, alpha=0.9)
        + geom_hline(yintercept=100, color="#9CA3AF", linetype="dotted", size=0.4)
        + scale_color_manual(values=PALETTE)
        + scale_x_datetime(date_breaks="5 years", date_labels="%Y")
        + labs(
            title="FX Spot Rates - Indexed to 100 at January 1995",
            y="Index (Jan 1995 = 100)",
            x="",
            color="Currency",
            caption="Values above 100 indicate foreign currency depreciation vs USD since Jan 1995.",
        )
        + theme_minimal()
        + theme(
            figure_size=(14, 6),
            legend_position="left",
            axis_text_x=element_text(rotation=0),
            plot_title=element_text(weight="bold"),
        )
    )


def plot_absolute_spot_levels(wide_df: pd.DataFrame) -> ggplot:
    """Figure 2: Full-history absolute spot levels by currency."""
    plot_df = (
        wide_df[CURRENCIES]
        .reset_index()
        .melt(id_vars="date", var_name="currency", value_name="rate_per_usd")
        .dropna()
    )
    plot_df["ymin"] = (
        plot_df.groupby("currency")["rate_per_usd"].transform("min") * 0.95
    )

    label_order = [f"{ccy} - {CURRENCY_NAMES[ccy]}" for ccy in CURRENCIES]
    label_palette = {
        f"{ccy} - {CURRENCY_NAMES[ccy]}": PALETTE[ccy] for ccy in CURRENCIES
    }
    plot_df["currency_label"] = plot_df["currency"].map(
        lambda c: f"{c} - {CURRENCY_NAMES[c]}"
    )
    plot_df["currency_label"] = pd.Categorical(
        plot_df["currency_label"], label_order, ordered=True
    )

    covid = pd.DataFrame(
        {"start": [pd.Timestamp("2020-02-15")], "end": [pd.Timestamp("2020-06-30")]}
    )

    return (
        ggplot(plot_df, aes("date", "rate_per_usd", color="currency_label"))
        + geom_rect(
            data=covid,
            mapping=aes(xmin="start", xmax="end", ymin=-np.inf, ymax=np.inf),
            inherit_aes=False,
            fill="#6B7280",
            alpha=0.10,
        )
        + geom_ribbon(
            aes(ymin="ymin", ymax="rate_per_usd", fill="currency_label"),
            alpha=0.12,
            color=None,
        )
        + geom_line(size=0.7)
        + facet_wrap("~currency_label", ncol=4, scales="free_y")
        + scale_color_manual(values=label_palette)
        + scale_fill_manual(values=label_palette)
        + scale_x_datetime(date_breaks="10 years", date_labels="'%y")
        + labs(
            title="FX Spot Rate Levels - Full History (Foreign Currency per USD)",
            y="per USD",
            x="",
            color="Currency",
            fill="Currency",
        )
        + theme_minimal()
        + theme(
            figure_size=(16, 7),
            legend_position="none",
            plot_title=element_text(weight="bold"),
            strip_text=element_text(weight="bold", size=8),
            axis_text_x=element_text(size=7),
            axis_text_y=element_text(size=7),
        )
    )


def plot_rolling_annualized_vol(log_ret: pd.DataFrame) -> ggplot:
    """Figure 3: 30-day rolling annualized volatility (2000-present)."""
    vol = (log_ret.rolling(30).std() * np.sqrt(252) * 100).loc[
        "2000-01-01":, CURRENCIES
    ]
    plot_df = (
        vol.reset_index()
        .melt(id_vars="date", var_name="currency", value_name="volatility")
        .dropna()
    )
    plot_df["currency"] = pd.Categorical(plot_df["currency"], CURRENCIES, ordered=True)

    return (
        ggplot(plot_df, aes("date", "volatility", color="currency"))
        + geom_rect(
            data=_shading_df(),
            mapping=aes(xmin="start", xmax="end", ymin=-np.inf, ymax=np.inf),
            inherit_aes=False,
            fill="#374151",
            alpha=0.07,
        )
        + geom_line(size=0.8, alpha=0.85)
        + scale_color_manual(values=PALETTE)
        + scale_x_datetime(date_breaks="5 years", date_labels="%Y")
        + labs(
            title="30-Day Rolling Annualized Volatility (2000-Present)",
            y="Annualized Volatility (%)",
            x="",
            color="Currency",
        )
        + theme_minimal()
        + theme(
            figure_size=(14, 6),
            legend_position="right",
            plot_title=element_text(weight="bold"),
        )
    )


def plot_log_return_correlation_heatmap(log_ret: pd.DataFrame) -> ggplot:
    """Figure 4: Daily log-return correlation matrix (2000-present)."""
    corr = log_ret.loc["2000-01-01":, CURRENCIES].corr()

    corr_long = corr.reset_index().melt(
        id_vars="currency", var_name="currency_col", value_name="corr"
    )
    corr_long = corr_long.rename(columns={"currency": "currency_row"})

    order = {ccy: idx for idx, ccy in enumerate(CURRENCIES)}
    corr_long = corr_long[
        corr_long.apply(
            lambda r: order[r["currency_row"]] >= order[r["currency_col"]], axis=1
        )
    ].copy()

    corr_long["currency_col"] = pd.Categorical(
        corr_long["currency_col"], CURRENCIES, ordered=True
    )
    corr_long["currency_row"] = pd.Categorical(
        corr_long["currency_row"], list(reversed(CURRENCIES)), ordered=True
    )
    corr_long["label"] = corr_long["corr"].map(lambda x: f"{x:.2f}")

    return (
        ggplot(corr_long, aes("currency_col", "currency_row", fill="corr"))
        + geom_tile(color="#E5E7EB", size=0.5)
        + geom_text(aes(label="label"), size=8)
        + scale_fill_gradient2(
            low="#3B82F6", mid="#F9FAFB", high="#EF4444", midpoint=0, limits=(-1, 1)
        )
        + labs(
            title="Daily Log-Return Correlation Matrix (2000-Present)",
            fill="Pearson r",
            x="",
            y="",
        )
        + theme_minimal()
        + theme(
            figure_size=(8, 7),
            plot_title=element_text(weight="bold"),
            panel_grid=element_blank(),
            axis_text_x=element_text(rotation=0),
            axis_text_y=element_text(rotation=0),
        )
    )


def plot_annual_return_heatmap(wide_df: pd.DataFrame) -> ggplot:
    """Figure 5: Annual FX returns vs USD (calendar heatmap)."""
    annual = wide_df[CURRENCIES].resample("YE").last()
    annual_ret = (-annual.pct_change() * 100).iloc[1:]
    annual_ret.index = annual_ret.index.year
    annual_ret = annual_ret.loc[1996:]

    plot_df = (
        annual_ret.reset_index(names="year")
        .melt(id_vars="year", var_name="currency", value_name="annual_return")
        .dropna()
    )
    plot_df["year"] = plot_df["year"].astype(str)
    plot_df["currency"] = pd.Categorical(
        plot_df["currency"], list(reversed(CURRENCIES)), ordered=True
    )
    plot_df["label"] = plot_df["annual_return"].map(lambda x: f"{x:.1f}")

    return (
        ggplot(plot_df, aes("year", "currency", fill="annual_return"))
        + geom_tile(color="#F3F4F6", size=0.3)
        + geom_text(aes(label="label"), size=6)
        + scale_fill_gradient2(
            low="#EF4444", mid="#F9FAFB", high="#22C55E", midpoint=0, limits=(-25, 25)
        )
        + labs(
            title="Annual Currency Returns vs USD (%)",
            subtitle="Green = strengthened vs USD; Red = weakened vs USD",
            fill="Annual Return (%)",
            x="",
            y="",
        )
        + theme_minimal()
        + theme(
            figure_size=(18, 4),
            plot_title=element_text(weight="bold"),
            axis_text_x=element_text(rotation=45, ha="right", size=7),
            axis_text_y=element_text(size=9),
            panel_grid=element_blank(),
        )
    )


def build_all_fx_spot_plots(df: pl.DataFrame) -> dict[str, ggplot]:
    """Create all FX spot exploration figures from a Polars frame."""
    _, wide_df, log_ret = prepare_fx_plot_data(df)
    return {
        "indexed_spot_rates": plot_indexed_spot_rates(wide_df),
        "absolute_spot_levels": plot_absolute_spot_levels(wide_df),
        "rolling_annualized_vol": plot_rolling_annualized_vol(log_ret),
        "log_return_corr_heatmap": plot_log_return_correlation_heatmap(log_ret),
        "annual_return_heatmap": plot_annual_return_heatmap(wide_df),
    }


def save_all_fx_spot_plots(output_dir: Path) -> list[Path]:
    """Generate and save all FX spot plots as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = build_all_fx_spot_plots(load_fx_spot())
    saved: list[Path] = []

    for name, plot in plots.items():
        path = output_dir / f"{name}.png"
        plot.save(path, dpi=120, verbose=False)
        saved.append(path)

    return saved


if __name__ == "__main__":
    saved_paths = save_all_fx_spot_plots(Path("notebooks/figures"))
    print("Saved plots:")
    for p in saved_paths:
        print(f"- {p}")
