import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_equity_returns(portfolio_log) -> pd.DataFrame:
    """
    Gets the equity series from the portfolio log produced from the
    initial backtesting with 0 hedge

    Args:
        portfolio_log (list[dict]): portfolio log with one important column:
            equity at a given date.

    Returns:
        pd.DataFrame: Dataframe with date as DateTimeIndex, and an equity column
    """

    # convert list of dictionaries to a dataframe
    df = pd.DataFrame(portfolio_log)

    # set date column to the index and keep only equity
    df.set_index("date", inplace=True, drop=True)
    df.index.name = None
    if "equity" not in df.columns:
        raise ValueError("equity column not found in portfolio_log")
    df = df[["equity"]]

    return df


def get_hedge_returns() -> pd.DataFrame:
    """
    gets the dollar hedged returns from dollar_etf.xlsx.

    Returns:
        pd.DataFrame with dates in DateTime Index and a single column
        that is the hedged returns
    """

    # read the dollar_etf.xlsx and set the index column to dates
    df = pd.read_excel(DATA_DIR / "dollar_etf.xlsx", index_col="Dates")

    # use equity column to compute our hedge returns
    df["hedge_returns"] = df["UUP US Equity"].pct_change()

    # drop it
    df.drop("UUP US Equity", axis=1, inplace=True)

    # return the dataframe
    return df


def compute_hedge_beta(portfolio_log) -> float:
    """

    Computes the hedged_beta

    Args:
        portfolio_log (list[dict]): list of outputted portfolio log from
        backtesting with no hedge

    Returns:
        float: our beta
    """

    # use helper functions to get the hedge and equity returns
    hedge_returns = get_hedge_returns()
    equity_returns = get_equity_returns(portfolio_log=portfolio_log)

    # convert the equity returns index to a Datetime index for merging
    equity_returns.index = pd.to_datetime(equity_returns.index)

    # merge on indexes in an inner merge
    df = hedge_returns.merge(
        equity_returns, left_index=True, right_index=True, how="inner"
    )

    # drop na for regressing
    clean_df = df[["equity", "hedge_returns"]].dropna()
    clean_df["equity_returns"] = clean_df["equity"].pct_change()
    clean_df = clean_df.dropna()

    # Check if you still have data left and then use np.polyfit to get the beta.
    if len(clean_df) < 2:
        return 0.0

    # convert hedge and equity returns to numpy
    x = clean_df["hedge_returns"].to_numpy()
    y = clean_df["equity_returns"].to_numpy()

    denom = np.sum(x**2)
    if denom == 0:
        return 0.0

    beta = np.sum(x * y) / denom
    return float(beta)


# p-value and r**2?
# 2022 - 2025 --> massive rates cut


def compute_hedge_beta_with_intercept(portfolio_log) -> float:
    """
    Computes the hedged_beta allowing for an intercept (alpha).

    Args:
        portfolio_log (list[dict]): list of outputted portfolio log from
        backtesting with no hedge

    Returns:
        float: our beta
    """

    # use helper functions to get the hedge and equity returns
    hedge_returns = get_hedge_returns()
    equity_returns = get_equity_returns(portfolio_log=portfolio_log)

    # convert the equity returns index to a Datetime index for merging
    equity_returns.index = pd.to_datetime(equity_returns.index)

    # merge on indexes in an inner merge
    df = hedge_returns.merge(
        equity_returns, left_index=True, right_index=True, how="inner"
    )

    # drop na for regressing
    clean_df = df[["equity", "hedge_returns"]].dropna()
    clean_df["equity_returns"] = clean_df["equity"].pct_change()
    clean_df = clean_df.dropna()

    # Check if you still have data left and then compute the beta with intercept.
    if len(clean_df) < 2:
        return 0.0

    # convert hedge and equity returns to numpy and mean-center them
    x = clean_df["hedge_returns"].to_numpy()
    y = clean_df["equity_returns"].to_numpy()
    x = x - x.mean()
    y = y - y.mean()

    denom = np.sum(x**2)
    if denom == 0:
        return 0.0

    beta = np.sum(x * y) / denom
    return float(beta)
