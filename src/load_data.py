"""Bloomberg data preparation utilities for the cross-asset FX analysis."""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

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
    "DEXJPUS": ("JPY", True),  # JPY per USD  → invert
    "DEXCAUS": ("CAD", True),  # CAD per USD  → invert
    "DEXMXUS": ("MXN", True),  # MXN per USD  → invert
    "DEXBZUS": ("BRL", True),  # BRL per USD  → invert
    "DEXSFUS": ("ZAR", True),  # ZAR per USD  → invert
    "DEXHKUS": ("HKD", True),  # HKD per USD  → invert
    "DEXINUS": ("INR", True),  # INR per USD  → invert
    "DEXKOUS": ("KRW", True),  # KRW per USD  → invert
    "DEXNOUS": ("NOK", True),  # NOK per USD  → invert
    "DEXSDUS": ("SEK", True),  # SEK per USD  → invert
    "DEXSIUS": ("SGD", True),  # SGD per USD  → invert
    "DEXSZUS": ("CHF", True),  # CHF per USD  → invert
    "DEXUSNZ": ("NZD", False),  # USD per NZD  → keep
}

# Edit these to your preferred market-close definitions                                                                                                                                                             
DEFAULT_MARKET_CLOSES = {                                                                                                                                                                                           
    "US": ("America/New_York", "16:00"),                                                                                                                                                                            
    "UK": ("Europe/London", "16:30"),                                                                                                                                                                               
    "Japan": ("Asia/Tokyo", "15:00"),                                                                                                                                                                               
    "Australia": ("Australia/Sydney", "16:00"),                                                                                                                                                                     
    "Canada": ("America/Toronto", "16:00"),                                                                                                                                                                         
    "Switzerland": ("Europe/Zurich", "17:30"),                                                                                                                                                                      
    "Hong Kong": ("Asia/Hong_Kong", "16:00"),                                                                                                                                                                       
    "Singapore": ("Asia/Singapore", "17:00"),                                                                                                                                                                       
    "India": ("Asia/Kolkata", "15:30"),                                                                                                                                                                             
    "South Korea": ("Asia/Seoul", "15:30"),                                                                                                                                                                         
    "Sweden": ("Europe/Stockholm", "17:30"),                                                                                                                                                                        
    "Norway": ("Europe/Oslo", "16:30"),                                                                                                                                                                             
    "New Zealand": ("Pacific/Auckland", "16:45"),                                                                                                                                                                   
    "South Africa": ("Africa/Johannesburg", "17:00"),                                                                                                                                                               
    "Brazil": ("America/Sao_Paulo", "17:00"),                                                                                                                                                                       
    "Mexico": ("America/Mexico_City", "15:00"),                                                                                                                                                                     
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

    df = pl.concat(frames).drop_nulls("rate_per_usd").sort(["date", "currency"])

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
            df = pd.read_csv(
                file, parse_dates=["observation_date"], index_col="observation_date"
            )
            df = df.sort_index()

            ticker = file.stem
            df = df.rename(columns={ticker: ticker})

            assert isinstance(
                fx.index, pd.DatetimeIndex
            ), "total_fx index is not DatetimeIndex"
            assert isinstance(
                df.index, pd.DatetimeIndex
            ), "country_fx index is not DatetimeIndex"

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
    yields = yields.rename(columns={"Dates": "date", "IRNI1M Curncy": "INRI1M Curncy", "KWNI1M Curncy": "KRWI1M Curncy", "BCNI1M Curncy": "BRLI1M Curncy"})
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
        fx = pd.read_csv(fx_path, index_col="date", parse_dates=["date"])
    else:
        fx = prepare_fx_spot_data(
            fx_data_dir=fx_data_dir,
            start_date=start_date,
            end_date=end_date,
        )

    # --- Carry ---
    carry_path = data_dir / "daily_carry.csv"
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


def read_signal_data(data_dir: Path = DATA_DIR):
    signals_df = pd.read_csv(data_dir / "rf_predictions.csv", index_col="date", parse_dates="date")
    return signals_df


def load_fx_futures_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    base = Path(data_dir) / "futures_1m"                                                                                                                                                                                                           
    files = list(base.rglob("*.csv"))                                                                                                                                                                                                
    if not files:                                                                                                                                                                                                                    
        return pd.DataFrame(columns=["ts_event", "close", "country"])                                                                                                                                                                
                                                                                                                                                                                                                                    
    frames = []                                                                                                                                                                                                                     
    for f in files:
        df = pd.read_csv(f, usecols=["ts_event", "close", "country"], parse_dates=["ts_event"])                                                                                                                                      
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def build_after_close_panel(                                                                                                                                                                                        
    combined_df: pd.DataFrame,                                                                                                                                                                                      
    market_closes: dict[str, tuple[str, str]] = DEFAULT_MARKET_CLOSES,                                                                                                                                              
    max_delay: str = "12h",
    data_dir: Path = DATA_DIR                                                                                                                                                                                
) -> pd.DataFrame:                                                                                                                                                                                                  
    """                                                                                                                                                                                                             
    Input: combined_df with columns ['ts_event', 'close', 'country'].                                                                                                                                               
    Output: wide df indexed by local market date, one column per country                                                                                                                                            
            (e.g., us_close), value = first futures price at/after local close.                                                                                                                                     
    """                                                                                                                                                                                                             
    df = combined_df.copy()                                                                                                                                                                                         
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")                                                                                                                                      
    df = df.dropna(subset=["ts_event", "close", "country"]).sort_values("ts_event")                                                                                                                                 
    max_delay_td = pd.Timedelta(max_delay)                                                                                                                                                                          
                                                                                                                                                                                                                    
    rows = []                                                                                                                                                                                                       
                                                                                                                                                                                                                    
    for country, (tz_name, close_hhmm) in market_closes.items():                                                                                                                                                    
        cdf = df[df["country"] == country].sort_values("ts_event")                                                                                                                                                  
        if cdf.empty:                                                                                                                                                                                               
            continue                                                                                                                                                                                                
                                                                                                                                                                                                                    
        ts = pd.DatetimeIndex(cdf["ts_event"])                                                                                                                                                                      
        close_vals = cdf["close"].to_numpy()                                                                                                                                                                        
                                                                                                                                                                                                                    
        local_ts = ts.tz_convert(ZoneInfo(tz_name))                                                                                                                                                                 
        start_date = local_ts.min().date()
        end_date = local_ts.max().date()                                                                                                                                                                            
                                                                                                                                                                                                                    
        hh, mm = map(int, close_hhmm.split(":"))                                                                                                                                                                    
                                                                                                                                                                                                                    
        for d in pd.date_range(start_date, end_date, freq="D"):                                                                                                                                                     
            cutoff_local = pd.Timestamp(                                                                                                                                                                            
                year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=ZoneInfo(tz_name)                                                                                                                     
            )                                                                                                                                                                                                       
            cutoff_utc = cutoff_local.tz_convert("UTC")                                                                                                                                                             
            next_cutoff_utc = cutoff_utc + pd.Timedelta(days=1)                                                                                                                                                     
                                                                                                                                                                                                                    
            i = ts.searchsorted(cutoff_utc, side="left")                                                                                                                                                            
            if i >= len(ts):                                                                                                                                                                                        
                continue                                                                                                                                                                                            
                                                                                                                                                                                                                    
            chosen_ts = ts[i]                                                                                                                                                                                       
            if chosen_ts < next_cutoff_utc and (chosen_ts - cutoff_utc) <= max_delay_td:                                                                                                                            
                rows.append(                                                                                                                                                                                        
                    {"date": pd.Timestamp(d.date()), "country": country, "close": close_vals[i]}                                                                                                                    
                )                                                                                                                                                                                                   
                                                                                                                                                                                                                    
    out = pd.DataFrame(rows)                                                                                                                                                                                        
    if out.empty:                                                                                                                                                                                                   
        return pd.DataFrame()                                                                                                                                                                                       
                                                                                                                                                                                                                    
    panel = out.pivot(index="date", columns="country", values="close").sort_index()                                                                                                                                 
    panel.columns = [f"{c.lower().replace(' ', '_')}_close" for c in panel.columns]

    out_path = data_dir / "fx_futures_panel.csv"
    panel.to_csv(out_path)

    return panel


if __name__ == "__main__":
    combined = load_fx_futures_data()
    after_close = build_after_close_panel(combined)