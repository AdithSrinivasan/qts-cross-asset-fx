
import pandas as pd
import numpy as np

def get_equity_returns(portfolio_log) -> pd.DataFrame:
    df = pd.DataFrame(portfolio_log)
    df.set_index("date", inplace=True, drop=True)
    df.index.name = None
    df.drop(["pl", "margin_used", "total_exposure", "target_asset_exposure"], axis=1, inplace=True)
    df.columns = ["equity"]
    df["equity_returns"] = df["equity"].pct_change()
    df.drop("equity", axis=1, inplace=True)
    df.head()
    return df

def get_hedge_returns():
    """
    """
    df = pd.read_excel("../data/dollar_etf.xlsx", index_col="Dates")
    df["hedge_returns"] = df["UUP US Equity"].pct_change()
    df.drop("UUP US Equity", axis=1, inplace=True)
    return df



def compute_hedge_beta(portfolio_log):
    hedge_returns = get_hedge_returns()
    equity_returns = get_equity_returns(portfolio_log=portfolio_log)
    equity_returns.index = pd.to_datetime(equity_returns.index)
    df = hedge_returns.merge(equity_returns, left_index=True, right_index=True, how='inner')
    print(df.head())
    clean_df = df[['equity_returns', 'hedge_returns']].dropna()

    # Check if you still have data left!
    if not clean_df.empty:
        beta, alpha = np.polyfit(clean_df['equity_returns'], clean_df['hedge_returns'], 1)
    return beta
