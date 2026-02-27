"""Bloomberg data preparation utilities for the cross-asset FX analysis."""

import pandas as pd


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
