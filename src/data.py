"""Bloomberg data preparation utilities for the cross-asset FX analysis."""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA_DIR = DATA_DIR / "fx_data"

# Maps FRED FX series code → (ISO currency code, needs_inversion)
#
# Target convention:
#     All series are standardized to USD per 1 unit of foreign currency.
#
# needs_inversion = True  → raw FRED series is foreign currency per USD,
#                            so we invert (1 / raw) to obtain USD per foreign.
# needs_inversion = False → raw FRED series is already USD per foreign,
#                            so we keep it as-is.
_USD_PER_FX_META: dict[str, tuple[str, bool]] = {
    "DEXUSUK": ("GBP", False),  # USD per GBP  → keep
    "DEXUSAL": ("AUD", False),  # USD per AUD  → keep
    "DEXJPUS": ("JPY", True),   # JPY per USD  → invert
    "DEXCAUS": ("CAD", True),   # CAD per USD  → invert
    "DEXMXUS": ("MXN", True),   # MXN per USD  → invert
    "DEXBZUS": ("BRL", True),   # BRL per USD  → invert
    "DEXSFUS": ("ZAR", True),   # ZAR per USD  → invert
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


def load_fx_spot(
    data_dir: Path = DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Return all FX spot rates normalised to foreign-currency-per-1-USD.

    Parameters
    ----------
    data_dir:
        Directory containing the FRED CSV files.  Defaults to the project's
        ``data/fx_data/`` folder.
    start_date:
        Optional start date filter in ``"YYYY-MM-DD"`` format (inclusive).
    end_date:
        Optional end date filter in ``"YYYY-MM-DD"`` format (inclusive).

    Returns
    -------
    pl.DataFrame
        Tidy frame sorted by date then currency with columns:
        ``date``, ``currency``, ``rate_per_usd``.
    """
    frames: list[pl.DataFrame] = []
    for series_code, (currency, invert) in _USD_PER_FX_META.items():
        path = data_dir / f"{series_code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        frames.append(_read_series(path, currency, invert))

    df = (
        pl.concat(frames)
        .drop_nulls("rate_per_usd")
        .sort(["date", "currency"])
    )

    if start_date is not None:
        df = df.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
    if end_date is not None:
        df = df.filter(pl.col("date") <= pl.lit(end_date).str.to_date())

    return df


def prepare_fx_spot_data(
    fx_data_dir: Path = FX_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
):
    # Load FX data
    fx = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    fx.set_index("date", inplace=True)

    country_fx_dfs = []

    for file in fx_data_dir.iterdir():
        if file.is_file():
            print("Loading FX data from...", file)
            df = pd.read_csv(file, parse_dates=["observation_date"], index_col="observation_date")
            df = df.sort_index()

            ticker = file.stem
            df = df.rename(columns={ticker: ticker})

            assert isinstance(fx.index, pd.DatetimeIndex), "total_fx index is not DatetimeIndex"
            assert isinstance(df.index, pd.DatetimeIndex), "country_fx index is not DatetimeIndex"

            country_fx_dfs.append(df)

    fx = pd.concat(country_fx_dfs, axis=1, sort=True)
    fx.reset_index(inplace=True)
    fx = fx.rename(columns={"observation_date": "date"}).set_index("date")
    fx = fx.sort_index()
    fx = fx.loc[start_date:end_date]

    # Rename columns from FRED codes to ISO currency codes, inverting where needed
    for series_code, (currency, invert) in _USD_PER_FX_META.items():
        if series_code in fx.columns:
            if invert:
                fx[series_code] = 1 / fx[series_code]
            fx = fx.rename(columns={series_code: currency})

    assert isinstance(fx.index, pd.DatetimeIndex)
    fx = fx.reindex(sorted(fx.columns), axis=1)
    fx.to_csv(DATA_DIR / "fx_data.csv")

    return fx


def prepare_fx_carry_data(
    data_dir: Path = DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
):

    yields = pd.read_excel(DATA_DIR / "fwd_yield_ann.xlsx", parse_dates=["Dates"])
    yields = yields.rename(columns={"Dates": "date"})
    yields = yields.set_index("date")
    yields.columns = yields.columns.str.replace("I1M Curncy", "", regex=False)
    yields = yields.reindex(sorted(yields.columns), axis=1)

    yields = yields.loc[start_date:end_date]

    carry = yields / 100 / 252
    carry.to_csv(DATA_DIR / "daily_carry.csv")

    return carry


def calculate_fx_excess_returns(
    data_dir: Path = DATA_DIR,
    fx_data_dir: Path = FX_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
):
    # --- Spot ---
    fx_path = data_dir / "fx_data.csv"
    if fx_path.is_file():
        fx = pd.read_csv(fx_path, index_col="date", parse_dates=True)
    else:
        fx = prepare_fx_spot_data(
            fx_data_dir=fx_data_dir,
            start_date=start_date,
            end_date=end_date,
        )

    # --- Carry ---
    carry_path = data_dir / "carry.csv"
    if carry_path.is_file():
        carry = pd.read_csv(carry_path, index_col="date", parse_dates=True)
    else:
        # IMPORTANT: this must be your carry builder, not spot
        # e.g. carry = prepare_fx_carry_data(data_dir=data_dir, start_date=start_date, end_date=end_date)
        carry = prepare_fx_carry_data(
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
        )

    # --- Spot log returns ---
    dlog_fx = np.log(fx).diff()

    # --- Align dates and currencies (critical) ---
    # Inner join ensures we only keep dates present in BOTH and drops the first diff NaN cleanly.
    common_cols = sorted(set(dlog_fx.columns).intersection(carry.columns))
    dlog_fx = dlog_fx[common_cols]
    carry = carry[common_cols]

    spot_and_carry = dlog_fx.join(carry, how="inner", lsuffix="_spot", rsuffix="_carry")

    # Build excess returns
    fx_ret = pd.DataFrame(index=spot_and_carry.index, columns=common_cols, dtype=float)
    fx_ret.index.name = "date"

    for c in common_cols:
        fx_ret[c] = spot_and_carry[f"{c}_spot"] + spot_and_carry[f"{c}_carry"]

    fx_ret = fx_ret.dropna()

    out_path = data_dir / "fx_returns_data.csv"
    fx_ret.to_csv(out_path)

    return fx_ret