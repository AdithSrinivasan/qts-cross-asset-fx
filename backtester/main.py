from backtester_engine import Backtester
from performance_output import plot_portfolio_history, print_portfolio_stats
import pandas as pd
from hedge import compute_hedge_beta

def main():
    # Get signals data
    train_predictions = pd.read_csv("../data/rf_train_predictions.csv", index_col="date")
    test_predictions = pd.read_csv("../data/rf_test_predictions.csv", index_col="date")
    entry_thresholds = pd.read_csv("../data/rf_thresholds.csv")
    exit_thresholds = pd.read_csv("../data/rf_exit_thresholds.csv")
    fx_contract_specs = pd.read_csv("../data/fx_contract_specs.csv")
    fx_futures_panel = pd.read_csv("../data/fx_futures_panel.csv", index_col="date")
    

    # Run backtesting simulation given parameters
    backtester = Backtester(return_predictions=train_predictions,
                            fx_futures_panel=fx_futures_panel,
                            entry_thresholds=entry_thresholds,
                            exit_thresholds=exit_thresholds,
                            fx_contract_specs=fx_contract_specs,
                            is_train=True,
                            contract_cost_fixed=0.07,
                            starting_equity=2_000_000,
                            leverage_multiplier=5.0,
                            hedge_positions=False)
    backtester.run_backtest()

    # Display results
    backtest_trade_log, backtest_portfolio_log = backtester.get_backtest_results()
    print(backtest_portfolio_log, backtest_trade_log)
    print_portfolio_stats(backtest_portfolio_log, backtest_trade_log)
    plot_portfolio_history(backtest_portfolio_log, backtest_trade_log)
    
    # get strategy returns. It'll give me a return strream I'll regress against hte hedge asset
    
    # get the hedge asset returns using dollar etf
    
    # compute hedge beta
    hedge_beta = compute_hedge_beta(backtest_portfolio_log)
    print(hedge_beta)
    
    # Now compute hedge PnL
    hedge_bt = Backtester(return_predictions=train_predictions,
                            fx_futures_panel=fx_futures_panel,
                            entry_thresholds=entry_thresholds,
                            exit_thresholds=exit_thresholds,
                            fx_contract_specs=fx_contract_specs,
                            is_train=True,
                            contract_cost_fixed=0.07,
                            starting_equity=2_000_000,
                            leverage_multiplier=5.0,
                            hedge_positions=True,
                            hedge_ratio=hedge_beta)


if __name__ == "__main__":
    main()
