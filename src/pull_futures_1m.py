"""Download minutely futures data for the dates and countries in rf_predictions.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import databento as db
import pandas as pd

from db_env_util import get_databento_api_key

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SIGNAL_PATH = DATA_DIR / "rf_predictions.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "futures_1m"
DEFAULT_KEY_PATH = Path(__file__).parent.parent / ".databento_api_key"
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_STYPE_IN = "continuous"
DEFAULT_CONTINUOUS_RANK = 0
DEFAULT_CONTRACT_SELECTION = "max-volume"
DEFAULT_MAX_RANKS = 4

# These defaults are CME FX futures roots. By default the script converts them
# to Databento continuous symbols such as 6B.c.0.
DEFAULT_COUNTRY_TO_SYMBOL = {
    "Mexico": "6M",
    "Brazil": "6L",
    "South Africa": "6Z",
    "Japan": "6J",
    "Australia": "6A",
    "Canada": "6C",
    "UK": "6B",
    "Hong Kong": "6H",
    "India": "IR",
    "South Korea": "KRW",
    "Norway": "NOK",
    "Sweden": "SEK",
    "Singapore": "SGD",
    "Switzerland": "6S",
    "New Zealand": "6N",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull Databento 1-minute futures bars for all countries and dates "
            "listed in data/rf_predictions.csv."
        )
    )
    parser.add_argument(
        "--signals-path",
        type=Path,
        default=DEFAULT_SIGNAL_PATH,
        help="CSV with a date column plus country columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where one CSV per country/date will be written.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Databento dataset name.",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Databento schema to request.",
    )
    parser.add_argument(
        "--stype-in",
        default=DEFAULT_STYPE_IN,
        help="Databento input symbology type.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help="Optional path to the Databento API key file. Defaults to ../.databento_api_key.",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping country names to Databento symbols.",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Optional subset of country columns to pull.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist on disk.",
    )
    parser.add_argument(
        "--continuous-rank",
        type=int,
        default=DEFAULT_CONTINUOUS_RANK,
        help="Continuous contract rank to request when --stype-in=continuous.",
    )
    parser.add_argument(
        "--contract-selection",
        choices=["front", "max-volume"],
        default=DEFAULT_CONTRACT_SELECTION,
        help="Select the front contract or the highest-volume contract for each day.",
    )
    parser.add_argument(
        "--max-ranks",
        type=int,
        default=DEFAULT_MAX_RANKS,
        help="Number of continuous ranks to compare when selecting by volume.",
    )
    return parser.parse_args()


def load_country_mapping(mapping_json: Path | None) -> dict[str, str]:
    if mapping_json is None:
        return dict(DEFAULT_COUNTRY_TO_SYMBOL)

    with mapping_json.open() as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise TypeError(f"Expected a JSON object in {mapping_json}, got {type(mapping).__name__}")

    return {str(country): str(symbol) for country, symbol in mapping.items()}


def load_signal_dates_and_countries(
    signals_path: Path,
    requested_countries: list[str] | None,
) -> tuple[pd.DatetimeIndex, list[str]]:
    if signals_path.exists():
        signals = pd.read_csv(signals_path, parse_dates=["date"])
    elif signals_path.name == "rf_predictions.csv":
        fallback_paths = [
            signals_path.with_name("rf_train_predictions.csv"),
            signals_path.with_name("rf_test_predictions.csv"),
        ]
        existing = [path for path in fallback_paths if path.exists()]
        if not existing:
            raise FileNotFoundError(
                f"Could not find {signals_path} or train/test fallback files in {signals_path.parent}"
            )
        frames = [pd.read_csv(path, parse_dates=["date"]) for path in existing]
        signals = pd.concat(frames, axis=0, ignore_index=True)
        print(
            f"{signals_path} not found. Using fallback file(s): "
            + ", ".join(str(path) for path in existing)
        )
    else:
        raise FileNotFoundError(f"Signals file not found: {signals_path}")

    if "date" not in signals.columns:
        raise ValueError(f"Expected a 'date' column in {signals_path}")

    available_countries = [column for column in signals.columns if column != "date"]
    countries = requested_countries or available_countries

    missing = sorted(set(countries) - set(available_countries))
    if missing:
        raise ValueError(
            f"Requested countries not found in {signals_path}: {', '.join(missing)}"
        )

    dates = pd.DatetimeIndex(signals["date"]).normalize().drop_duplicates().sort_values()
    return dates, countries


def build_request_symbol(symbol: str, stype_in: str, continuous_rank: int) -> str:
    if stype_in != "continuous":
        return symbol

    if ".c." in symbol:
        return symbol

    return f"{symbol}.c.{continuous_rank}"


def fetch_range_df(
    client: db.Historical,
    *,
    dataset: str,
    schema: str,
    stype_in: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=[symbol],
        stype_in=stype_in,
        start=start,
        end=end,
    )
    return data.to_df()


def choose_contract_by_volume(
    client: db.Historical,
    *,
    base_symbol: str,
    day: pd.Timestamp,
    dataset: str,
    schema: str,
    stype_in: str,
    max_ranks: int,
) -> tuple[str | None, pd.DataFrame]:
    if stype_in != "continuous":
        raise ValueError("contract-selection=max-volume requires --stype-in=continuous")

    start = day.isoformat()
    end = (day + pd.Timedelta(days=1)).isoformat()

    best_symbol = None
    best_df = pd.DataFrame()
    best_volume = -1.0

    for rank in range(max_ranks):
        request_symbol = build_request_symbol(base_symbol, stype_in, rank)
        df = fetch_range_df(
            client,
            dataset=dataset,
            schema=schema,
            stype_in=stype_in,
            symbol=request_symbol,
            start=start,
            end=end,
        )
        if df.empty:
            continue

        total_volume = float(df["volume"].sum()) if "volume" in df.columns else 0.0
        if total_volume > best_volume:
            best_symbol = request_symbol
            best_df = df
            best_volume = total_volume

    return best_symbol, best_df


def normalize_day_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None).normalize()
    return index.normalize()


def fetch_rank_day_groups(
    client: db.Historical,
    *,
    base_symbol: str,
    dates: pd.DatetimeIndex,
    dataset: str,
    schema: str,
    stype_in: str,
    max_ranks: int,
) -> dict[int, dict[pd.Timestamp, pd.DataFrame]]:
    if stype_in != "continuous":
        raise ValueError("contract-selection=max-volume requires --stype-in=continuous")
    if len(dates) == 0:
        return {}

    start = dates.min().isoformat()
    end = (dates.max() + pd.Timedelta(days=1)).isoformat()
    rank_to_day_groups: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {}

    for rank in range(max_ranks):
        request_symbol = build_request_symbol(base_symbol, stype_in, rank)
        df = fetch_range_df(
            client,
            dataset=dataset,
            schema=schema,
            stype_in=stype_in,
            symbol=request_symbol,
            start=start,
            end=end,
        )
        if df.empty:
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected DatetimeIndex for {request_symbol}, got {type(df.index).__name__}"
            )

        df = df.copy()
        df["_day"] = normalize_day_index(df.index)
        day_groups = {
            day: group.drop(columns=["_day"])
            for day, group in df.groupby("_day", sort=False)
        }
        rank_to_day_groups[rank] = day_groups

    return rank_to_day_groups


def fetch_and_store_country_max_volume(
    client: db.Historical,
    *,
    country: str,
    symbol: str,
    dates: pd.DatetimeIndex,
    output_dir: Path,
    dataset: str,
    schema: str,
    stype_in: str,
    max_ranks: int,
    skip_existing: bool,
) -> None:
    country_dir = output_dir / country.replace(" ", "_")
    country_dir.mkdir(parents=True, exist_ok=True)

    rank_to_day_groups = fetch_rank_day_groups(
        client,
        base_symbol=symbol,
        dates=dates,
        dataset=dataset,
        schema=schema,
        stype_in=stype_in,
        max_ranks=max_ranks,
    )

    for day in dates:
        output_path = country_dir / f"{day:%Y-%m-%d}.csv"
        if skip_existing and output_path.exists():
            print(f"Skipping existing file: {output_path}")
            continue

        best_symbol = None
        best_df = pd.DataFrame()
        best_volume = -1.0

        for rank in range(max_ranks):
            day_groups = rank_to_day_groups.get(rank)
            if day_groups is None:
                continue
            day_df = day_groups.get(day)
            if day_df is None or day_df.empty:
                continue

            total_volume = float(day_df["volume"].sum()) if "volume" in day_df.columns else 0.0
            if total_volume > best_volume:
                best_symbol = build_request_symbol(symbol, stype_in, rank)
                best_df = day_df
                best_volume = total_volume

        if best_symbol is None or best_df.empty:
            print(
                f"No data returned for {country} "
                f"({symbol}, stype_in={stype_in}) on {day:%Y-%m-%d}"
            )
            continue

        out_df = best_df.copy()
        out_df["selected_symbol"] = best_symbol
        out_df["country"] = country
        out_df.to_csv(output_path)
        print(f"Wrote {len(out_df):,} rows for {best_symbol} to {output_path}")


def fetch_and_store_country_day(
    client: db.Historical,
    *,
    country: str,
    symbol: str,
    day: pd.Timestamp,
    output_dir: Path,
    dataset: str,
    schema: str,
    stype_in: str,
    contract_selection: str,
    continuous_rank: int,
    max_ranks: int,
    skip_existing: bool,
) -> None:
    country_dir = output_dir / country.replace(" ", "_")
    country_dir.mkdir(parents=True, exist_ok=True)

    output_path = country_dir / f"{day:%Y-%m-%d}.csv"
    if skip_existing and output_path.exists():
        print(f"Skipping existing file: {output_path}")
        return

    if contract_selection == "max-volume":
        request_symbol, df = choose_contract_by_volume(
            client,
            base_symbol=symbol,
            day=day,
            dataset=dataset,
            schema=schema,
            stype_in=stype_in,
            max_ranks=max_ranks,
        )
        if request_symbol is None:
            print(
                f"No data returned for {country} "
                f"({symbol}, stype_in={stype_in}) on {day:%Y-%m-%d}"
            )
            return
    else:
        start = day.isoformat()
        end = (day + pd.Timedelta(days=1)).isoformat()
        request_symbol = build_request_symbol(symbol, stype_in, continuous_rank)
        df = fetch_range_df(
            client,
            dataset=dataset,
            schema=schema,
            stype_in=stype_in,
            symbol=request_symbol,
            start=start,
            end=end,
        )

    if df.empty:
        print(
            f"No data returned for {country} "
            f"({request_symbol}, stype_in={stype_in}) on {day:%Y-%m-%d}"
        )
        return

    df = df.copy()
    df["selected_symbol"] = request_symbol
    df["country"] = country
    df.to_csv(output_path)
    print(f"Wrote {len(df):,} rows for {request_symbol} to {output_path}")


def main() -> None:
    args = parse_args()

    dates, countries = load_signal_dates_and_countries(args.signals_path, args.countries)
    country_to_symbol = load_country_mapping(args.mapping_json)

    unmapped = [country for country in countries if country not in country_to_symbol]
    if unmapped:
        raise ValueError(
            "Missing symbol mappings for: "
            + ", ".join(unmapped)
            + ". Use --mapping-json to provide them."
        )

    client = db.Historical(get_databento_api_key(args.api_key_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Pulling {len(dates)} dates for {len(countries)} countries "
        f"from {args.signals_path} into {args.output_dir}"
    )

    for country in countries:
        symbol = country_to_symbol[country]
        if args.contract_selection == "max-volume":
            fetch_and_store_country_max_volume(
                client,
                country=country,
                symbol=symbol,
                dates=dates,
                output_dir=args.output_dir,
                dataset=args.dataset,
                schema=args.schema,
                stype_in=args.stype_in,
                max_ranks=args.max_ranks,
                skip_existing=args.skip_existing,
            )
            continue

        for day in dates:
            fetch_and_store_country_day(
                client,
                country=country,
                symbol=symbol,
                day=day,
                output_dir=args.output_dir,
                dataset=args.dataset,
                schema=args.schema,
                stype_in=args.stype_in,
                contract_selection=args.contract_selection,
                continuous_rank=args.continuous_rank,
                max_ranks=args.max_ranks,
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    main()
