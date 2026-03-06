from backtester_engine import Backtester
from performance_output import plot_portfolio_history, print_portfolio_stats
import pandas as pd
from src.load_data import load_fx_futures_data 


def main():
    # Get signals data
    train_predictions = pd.read_csv("../data/rf_train_predictions.csv")
    test_predictions = pd.read_csv("../data/rf_test_predictions.csv")
    entry_thresholds = pd.read_csv("../data/rf_thresholds.csv")
    exit_thresholds = pd.read_csv("../data/rf_exit_thresholds")
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
    backtest_trade_log, backtest_equity_log = backtester.get_backtest_results()
    print_portfolio_stats(equity_log=backtest_equity_log)
    # print_portfolio_stats(backtest_value_log)
    # plot_portfolio_history(backtest_value_log)


if __name__ == "__main__":
    main()
