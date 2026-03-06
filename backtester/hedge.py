
import pandas as pd
import numpy as np

def get_equity_returns(portfolio_log) -> pd.DataFrame:
    df = pd.DataFrame(portfolio_log)
    df.set_index("date", inplace=True, drop=True)
    df.index.name = None
    df.drop(["pl", "margin_used", "total_exposure", "target_asset_exposure"], axis=1, inplace=True)
    df.columns = ["returns"]
    df["returns"] = df["returns"].pct_change()
    df.head()
    return df

def get_hedge_returns():
    """
    """
    df = pd.read_excel("../data/dollar_etf.xlsx", index_col="Dates")
    df["returns"] = df["UUP US Equity"].pct_change()
    df.drop("UUP US Equity", axis=1, inplace=True)
    return df



def compute_hedge_beta(portfolio_log):
    hedge_returns = get_hedge_returns()
    equity_returns = get_equity_returns(portfolio_log=portfolio_log)
    pass