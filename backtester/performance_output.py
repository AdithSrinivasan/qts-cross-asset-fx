import math
import matplotlib.pyplot as plt


def print_portfolio_stats(equity_history):
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

    # downside deviation for Sortino
    downside_returns = [r for r in returns if r < 0]
    if len(downside_returns) > 1:
        downside_mean = sum(downside_returns) / len(downside_returns)
        downside_variance = sum((r - downside_mean) ** 2 for r in downside_returns) / (len(downside_returns) - 1)
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

    # skewness and excess kurtosis
    skew = 0.0
    kurt = 0.0

    if len(returns) >= 3 and vol > 0:
        n = len(returns)
        m = avg_return

        m2 = sum((r - m) ** 2 for r in returns) / n
        m3 = sum((r - m) ** 3 for r in returns) / n
        m4 = sum((r - m) ** 4 for r in returns) / n

        if m2 > 0:
            skew = m3 / (m2 ** 1.5)
            kurt = (m4 / (m2 ** 2)) - 3.0

    win_rate = sum(1 for r in returns if r > 0) / len(returns)
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
    print(f"Skewness:            {skew:.4f}")
    print(f"Excess Kurtosis:     {kurt:.4f}")
    print(f"Win Rate:            {100 * win_rate:.2f}%")
    print(f"Best Period Return:  {100 * best_period:.2f}%")
    print(f"Worst Period Return: {100 * worst_period:.2f}%")
    print(f"Num Observations:    {len(equity_history)}")

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt


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
        timestamps = [row[time_key] for row in equity_history]
    else:
        # fallback: just use index if no timestamp exists
        timestamps = list(range(len(equity_history)))

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

    plt.title("Portfolio Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()