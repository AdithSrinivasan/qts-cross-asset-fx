"""Plotnine charts for the cross-asset FX analysis notebook."""

from __future__ import annotations

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_line,
    ggplot,
    labs,
    theme,
    theme_gray,
    theme_minimal,
)


def plot_rebased_equity_indices(df: pd.DataFrame) -> ggplot:
    """Line chart of equity indices rebased to 100 at the first observation."""
    if df.index.name in ("date", "Dates") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Dates" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Dates": "date"})

    df["date"] = pd.to_datetime(df["date"])

    df_rebased = df.copy()
    for col in df.columns:
        if col != "date":
            df_rebased[col] = df[col] / df[col].iloc[0] * 100

    df_long = df_rebased.melt(id_vars="date", var_name="Index", value_name="Level")
    df_long["Level"] = pd.to_numeric(df_long["Level"], errors="coerce")
    df_long = df_long.dropna(subset=["date", "Level"])

    return (
        ggplot(df_long, aes(x="date", y="Level", color="Index"))
        + geom_line(size=1)
        + theme_minimal()
        + labs(
            title="Equity Indices Over Time",
            x="Date",
            y="Rebased Index Level (Base = 100)",
        )
        + theme(
            figure_size=(12, 6),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
        )
    )


def plot_cds_data(df: pd.DataFrame) -> list[ggplot]:
    """Return a list of CDS plots: one combined view, then one per series."""
    if df.index.name in ("date", "Dates") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Dates" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Dates": "date"})

    df["date"] = pd.to_datetime(df["date"])

    df_long = df.melt(id_vars="date", var_name="CDS", value_name="Spread")
    df_long["Spread"] = pd.to_numeric(df_long["Spread"], errors="coerce")
    df_long = df_long.dropna()

    plots: list[ggplot] = [
        ggplot(df_long, aes("date", "Spread", color="CDS"))
        + geom_line(size=1)
        + theme_gray()
        + labs(title="CDS Spreads Over Time", x="Date", y="Spread (bps)")
        + theme(figure_size=(14, 6))
    ]

    for col in df.columns:
        if col != "date":
            plots.append(
                ggplot(df, aes("date", col))
                + geom_line(size=1, color="steelblue")
                + theme_gray()
                + labs(title=f"{col} Over Time", x="Date", y="Spread (bps)")
                + theme(figure_size=(12, 4))
            )

    return plots
