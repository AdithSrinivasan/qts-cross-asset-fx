"""Bloomberg data preparation utilities for the cross-asset FX analysis."""

import pandas as pd
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "fx_data"

# Maps FRED series code → (ISO currency code, invert_flag)
# invert=True  → raw file is USD per FOREIGN, so rate_per_usd = 1 / raw
# invert=False → raw file is FOREIGN per USD, so rate_per_usd = raw
_SERIES_META: dict[str, tuple[str, bool]] = {
    "DEXUSUK": ("GBP", True),   # USD per GBP  → invert
    "DEXUSAL": ("AUD", True),   # USD per AUD  → invert
    "DEXJPUS": ("JPY", False),  # JPY per USD
    "DEXCAUS": ("CAD", False),  # CAD per USD
    "DEXMXUS": ("MXN", False),  # MXN per USD
    "DEXBZUS": ("BRL", False),  # BRL per USD
    "DEXSFUS": ("ZAR", False),  # ZAR per USD
}


def _read_series(path: Path, currency: str, invert: bool) -> pl.DataFrame:
    series_code = path.stem
    df = pl.read_csv(path, try_parse_dates=True).rename(
        {"observation_date": "date", series_code: "rate_per_usd"}
    )
    # FRED encodes missing observations as "." — cast safely
    df = df.with_columns(pl.col("rate_per_usd").cast(pl.Float64, strict=False))

    if invert:
        df = df.with_columns((1.0 / pl.col("rate_per_usd")).alias("rate_per_usd"))

    return df.with_columns(pl.lit(currency).alias("currency")).select(
        ["date", "currency", "rate_per_usd"]
    )


def prepare_bbg_data(
    data: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    """Filter, forward-fill and clean a Bloomberg Excel export.

    Parameters
    ----------
    data:
        Raw DataFrame with a 'Dates' column (or 'date' if already renamed).
    start_date:
        Exclusive lower bound as an ISO date string (e.g. '2024-01-01').
    end_date:
        Exclusive upper bound as an ISO date string.

    Returns
    -------
    pd.DataFrame
        Cleaned frame indexed by date.
    """
    data = data.copy()
    data = data.rename(columns={"Dates": "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")
    data = data[
        (data.index > pd.to_datetime(start_date))
        & (data.index < pd.to_datetime(end_date))
    ]
    data = data.ffill()
    data = data.dropna()
    return data


def _read_series(path: Path, currency: str, invert: bool) -> pl.DataFrame:
    series_code = path.stem
    df = pl.read_csv(path, try_parse_dates=True).rename(
        {"observation_date": "date", series_code: "rate_per_usd"}
    )
    # FRED encodes missing observations as "." — cast safely
    df = df.with_columns(pl.col("rate_per_usd").cast(pl.Float64, strict=False))

    if invert:
        df = df.with_columns((1.0 / pl.col("rate_per_usd")).alias("rate_per_usd"))

    return df.with_columns(pl.lit(currency).alias("currency")).select(
        ["date", "currency", "rate_per_usd"]
    )


def load_fx_spot(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Return all FX spot rates normalised to foreign-currency-per-1-USD.

    Parameters
    ----------
    data_dir:
        Directory containing the FRED CSV files.  Defaults to the project's
        ``data/fx_data/`` folder.

    Returns
    -------
    pl.DataFrame
        Tidy frame sorted by date then currency with columns:
        ``date``, ``currency``, ``rate_per_usd``.
    """
    frames: list[pl.DataFrame] = []
    for series_code, (currency, invert) in _SERIES_META.items():
        path = data_dir / f"{series_code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        frames.append(_read_series(path, currency, invert))

    return (
        pl.concat(frames)
        .drop_nulls("rate_per_usd")
        .sort(["date", "currency"])
    )

