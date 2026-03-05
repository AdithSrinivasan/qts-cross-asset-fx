from backtester_engine import Backtester
from performance_output import plot_portfolio_history, print_portfolio_stats
from data import read_signal_data


def main():
    # Get signals data
    signals_df = read_signal_data()

    # Run backtesting simulation given parameters
    backtester = Backtester(signals_df=signals_df)
    backtester.run_backtest()

    # Display results
    backtest_trade_log, backtest_value_log = backtester.get_backtest_results()
    # print_portfolio_stats(backtest_value_log)
    # plot_portfolio_history(backtest_value_log)


if __name__ == "__main__":
    main()
