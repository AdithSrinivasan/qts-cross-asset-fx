from backtester_engine import Backtester
from performance_output import plot_portfolio_history, print_portfolio_stats
import pandas as pd


def main():
    # Get signals data
    train_predictions = pd.read_csv("../data/rf_train_predictions.csv")
    test_predictions = pd.read_csv("../data/rf_test_predictions.csv")
    entry_thresholds = pd.read_csv("../data/rf_thresholds.csv")
    exit_thresholds = pd.read_csv("../data/rf_exit_thresholds")
    

    # Run backtesting simulation given parameters
    backtester = Backtester(train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            entry_thresholds=entry_thresholds,
                            exit_thresholds=exit_thresholds)
    backtester.run_backtest()

    # Display results
    backtest_trade_log, backtest_value_log = backtester.get_backtest_results()
    # print_portfolio_stats(backtest_value_log)
    # plot_portfolio_history(backtest_value_log)


if __name__ == "__main__":
    main()
