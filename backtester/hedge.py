
import pandas as pd
import numpy as np

def get_equity_returns(portfolio_log) -> pd.DataFrame:
    """
    Gets the equity returns from the portfolio log produced from the 
    initial backtesting with 0 hedge

    Args:
        portfolio_log (list[dict]): portfolio log with one important column:
            equity at a given date.

    Returns:
        pd.DataFrame: Dataframe with date as  DateTimeIndex, and an equity_return column
    """
    
    # convert list of dictionaries to a dataframe
    df = pd.DataFrame(portfolio_log)
    
    # set date column to the index and drop irrelevant non-equity columns
    df.set_index("date", inplace=True, drop=True)
    df.index.name = None
    df.drop(["pl", "margin_used", "total_exposure", "target_asset_exposure"], axis=1, inplace=True)
    
    # rename remaining column to equity
    df.columns = ["equity"]
    
    # calculate equity returns, and drop the equity column you used to compute those returns
    df["equity_returns"] = df["equity"].pct_change()
    df.drop("equity", axis=1, inplace=True)
    
    # return the head
    return df

def get_hedge_returns() -> pd.DataFrame:
    """
    gets the dollar hedged returns from dollar_etf.xlsx.
    
    Returns:
        pd.DataFrame with dates in DateTime Index and a single column
        that is the hedged returns
    """
    
    # read the dollar_etf.xlsx and set the index column to dates
    df = pd.read_excel("../data/dollar_etf.xlsx", index_col="Dates")
    
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
    df = hedge_returns.merge(equity_returns, left_index=True, right_index=True, how='inner')
    
    # drop na for regressing
    clean_df = df[['equity_returns', 'hedge_returns']].dropna()

    # Check if you still have data left and then use np.polyfit to get the beta.
    if not clean_df.empty:
        beta, alpha = np.polyfit(clean_df['equity_returns'], clean_df['hedge_returns'], 1)
    # return our found beta
    return beta
