import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def print_portfolio_stats(equity_history, trades=None):
    """
    equity_history: list of dicts like
    [
        {"ts": ..., "equity": ...},
        ...
    ]
    """

    if not equity_history:
        print("No equity history found.")
        return

    if len(equity_history) < 2:
        print("Not enough data to compute portfolio statistics.")
        print(f"Start Equity: {float(equity_history[0]['equity']):.2f}")
        print(f"End Equity:   {float(equity_history[0]['equity']):.2f}")
        return

    equities = [float(row["equity"]) for row in equity_history]

    returns = []
    for i in range(1, len(equities)):
        prev_eq = equities[i - 1]
        curr_eq = equities[i]

        if prev_eq != 0:
            returns.append((curr_eq / prev_eq) - 1.0)

    if not returns:
        print("Could not compute returns from equity history.")
        return

    start_equity = equities[0]
    end_equity = equities[-1]
    total_return = (end_equity / start_equity) - 1.0 if start_equity != 0 else 0.0

    # mean return
    avg_return = sum(returns) / len(returns)

    # sample standard deviation
    if len(returns) > 1:
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        vol = math.sqrt(variance)
    else:
        vol = 0.0

    sharpe = 0.0
    if vol > 0:
        sharpe = (avg_return / vol) * math.sqrt(252)

    # downside deviation for Sortino (target return = 0)
    downside_returns = [r for r in returns if r < 0]
    if len(downside_returns) >= 1:
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance)
    else:
        downside_dev = 0.0

    sortino = 0.0
    if downside_dev > 0:
        sortino = (avg_return / downside_dev) * math.sqrt(252)

    # max drawdown
    running_peak = equities[0]
    max_drawdown = 0.0

    for eq in equities:
        if eq > running_peak:
            running_peak = eq

        if running_peak != 0:
            drawdown = (eq / running_peak) - 1.0
            max_drawdown = min(max_drawdown, drawdown)

    # if len(returns) >= 3 and vol > 0:
    #     n = len(returns)
    #     m = avg_return

    #     m2 = sum((r - m) ** 2 for r in returns) / n
    #     m3 = sum((r - m) ** 3 for r in returns) / n
    #     m4 = sum((r - m) ** 4 for r in returns) / n

    #     if m2 > 0:
    #         skew = m3 / (m2**1.5)
    #         kurt = (m4 / (m2**2)) - 3.0

    active_returns = [r for r in returns if r != 0]
    win_rate = sum(1 for r in active_returns if r > 0) / len(active_returns) if active_returns else 0.0
    wins = [r for r in active_returns if r > 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    losses = [r for r in active_returns if r < 0]
    avg_loss = sum(losses) / len(losses) if losses else 0
    best_period = max(returns)
    worst_period = min(returns)

    annualized_return = (1.0 + avg_return) ** 252 - 1.0
    annualized_vol = vol * math.sqrt(252)

    print("Portfolio Statistics")
    print("-" * 40)
    print(f"Start Equity:        {start_equity:,.2f}")
    print(f"End Equity:          {end_equity:,.2f}")
    print(f"Total Return:        {100 * total_return:.2f}%")
    print(f"Annualized Return:   {100 * annualized_return:.2f}%")
    print(f"Annualized Vol:      {100 * annualized_vol:.2f}%")
    print(f"Sharpe Ratio:        {sharpe:.4f}")
    print(f"Sortino Ratio:       {sortino:.4f}")
    print(f"Max Drawdown:        {100 * max_drawdown:.2f}%")
    print(f"Win Rate:            {100 * win_rate:.2f}%")
    print(f"Average Win:         {100 * avg_win:.2f}%")
    print(f"Average Loss:        {100 * avg_loss:.2f}%")
    print(f"Best Period Return:  {100 * best_period:.2f}%")
    print(f"Worst Period Return: {100 * worst_period:.2f}%")
    print(f"Num Observations:    {len(equity_history)}")
    print(f"Total Trades:        {len(trades) if trades is not None else 0}")
    if "total_trading_fees" in equity_history[-1]:
        print(
            f"Total Trading Fees:  {float(equity_history[-1]['total_trading_fees']):,.2f}"
        )

def plot_free_margin_vs_total_maintenance_margin(equity_history):
    if not equity_history:
        print("No equity history found.")
        return

    sample = equity_history[0]
    time_key = None
    for key in ["ts", "date", "timestamp"]:
        if key in sample:
            time_key = key
            break

    required_keys = ("equity", "margin_used", "total_maintenance_margin")
    if not all(all(k in row for k in required_keys) for row in equity_history):
        print(
            "Skipping free margin vs maintenance margin plot: required fields missing."
        )
        return

    if time_key is not None:
        timestamps = pd.to_datetime(
            [row[time_key] for row in equity_history], errors="coerce"
        )
    else:
        timestamps = list(range(len(equity_history)))

    free_margin = [float(row["equity"]) - float(row["margin_used"]) for row in equity_history]
    total_maintenance_margin = [float(row["total_maintenance_margin"]) for row in equity_history]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, free_margin, label="Free Margin")
    plt.plot(timestamps, total_maintenance_margin, label="Total Maintenance Margin")
    plt.title("Free Margin vs Total Maintenance Margin")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    if time_key is not None:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=50)
        formatter = mdates.ConciseDateFormatter(locator)
        ax = plt.gca()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()


def plot_portfolio_history(equity_history, trades=None):
    if not equity_history:
        print("No equity history found.")
        return

    # figure out which time key exists
    sample = equity_history[0]
    time_key = None
    for key in ["ts", "date", "timestamp"]:
        if key in sample:
            time_key = key
            break

    equities = [float(row["equity"]) for row in equity_history]

    if time_key is not None:
        timestamps = pd.to_datetime(
            [row[time_key] for row in equity_history], errors="coerce"
        )
    else:
        # fallback: just use index if no timestamp exists
        timestamps = list(range(len(equity_history)))

    def _format_time_axis(axes):
        if hasattr(axes, "flat"):
            axes = list(axes.flat)
        elif not isinstance(axes, (list, tuple)):
            axes = [axes]
        locator = mdates.AutoDateLocator(minticks=6, maxticks=50)
        formatter = mdates.ConciseDateFormatter(locator)
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equities, label="Portfolio Equity")

    if trades and time_key is not None:
        equity_by_ts = {row[time_key]: float(row["equity"]) for row in equity_history}

        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for trade in trades:
            trade_ts = None
            for key in ["ts", "date", "timestamp"]:
                if key in trade:
                    trade_ts = trade[key]
                    break

            if trade_ts is None or trade_ts not in equity_by_ts:
                continue

            side = str(trade.get("side", "")).upper()

            if side == "BUY":
                buy_x.append(trade_ts)
                buy_y.append(equity_by_ts[trade_ts])
            elif side == "SELL":
                sell_x.append(trade_ts)
                sell_y.append(equity_by_ts[trade_ts])

        if buy_x:
            plt.scatter(buy_x, buy_y, marker="^", label="Buy Trades")
        if sell_x:
            plt.scatter(sell_x, sell_y, marker="v", label="Sell Trades")

    ax = plt.gca()
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    _format_time_axis(ax)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # Plot daily PnL and cumulative PnL
    if all("pl" in row for row in equity_history):
        daily_pl = [float(row["pl"]) for row in equity_history]
        cumulative_pl = []
        running_pl = 0.0
        for pnl in daily_pl:
            running_pl += pnl
            cumulative_pl.append(running_pl)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(timestamps, daily_pl, label="Daily PnL")
        axes[0].set_title("Daily PnL")
        axes[0].set_ylabel("PnL")
        axes[0].legend()

        axes[1].plot(timestamps, cumulative_pl, label="Cumulative Daily PnL")
        axes[1].set_title("Cumulative Daily PnL")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("PnL")
        axes[1].legend()
        _format_time_axis(axes)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping PnL plots: 'pl' not found in equity history.")

    # Plot free margin against maintenance margin
    plot_free_margin_vs_total_maintenance_margin(equity_history)

    # Plot total exposure and target total exposure
    if all(
        ("total_exposure" in row and "target_total_exposure" in row)
        for row in equity_history
    ):
        total_exposure = [float(row["total_exposure"]) for row in equity_history]
        target_exposure = [
            float(row["target_total_exposure"]) for row in equity_history
        ]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, total_exposure, label="Total Exposure")
        plt.plot(timestamps, target_exposure, label="Target Total Exposure")
        plt.title("Total vs Target Total Exposure")
        plt.xlabel("Time")
        plt.ylabel("Exposure")
        plt.legend()
        _format_time_axis(plt.gca())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print(
            "Skipping exposure plots: 'total_exposure' or 'target_total_exposure' missing."
        )

    # Plot total trading fees over time
    if all("total_trading_fees" in row for row in equity_history):
        total_trading_fees = [
            float(row["total_trading_fees"]) for row in equity_history
        ]
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, total_trading_fees, label="Total Trading Fees")
        plt.title("Total Trading Fees Over Time")
        plt.xlabel("Time")
        plt.ylabel("Fees")
        plt.legend()
        _format_time_axis(plt.gca())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping trading fees plot: 'total_trading_fees' missing.")

    # Plot number of open positions over time
    if all("num_positions" in row for row in equity_history):
        num_positions = [int(row["num_positions"]) for row in equity_history]
        plt.figure(figsize=(12, 4))
        plt.step(timestamps, num_positions, where="post", label="Open Positions")
        plt.title("Number of Open Positions Over Time")
        plt.xlabel("Time")
        plt.ylabel("Positions")
        plt.legend()
        _format_time_axis(plt.gca())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping positions plot: 'num_positions' missing.")

    # Plot cumulative PnL per country (return decomposition)
    if all("country_pnl" in row for row in equity_history):
        countries = list(equity_history[0]["country_pnl"].keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        for country in countries:
            daily = [float(row["country_pnl"].get(country, 0.0)) for row in equity_history]
            cumulative = []
            running = 0.0
            for v in daily:
                running += v
                cumulative.append(running)
            ax.plot(timestamps, cumulative, label=country)
        ax.set_title("Cumulative PnL by Country")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        _format_time_axis(ax)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping country PnL plot: 'country_pnl' missing.")
