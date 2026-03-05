import pandas as pd

class Backtester:
    def __init__(self, signals_df, trading_cost=0.01, starting_cash=100000, q_trading_thres=0.2, fx_look_ahead=1, leverage_multiplier=5.0):
        self.signals_df = signals_df.copy()
        self.trading_cost = trading_cost
        self.q_trading_thres = q_trading_thres
        self.fx_look_ahead = fx_look_ahead
        self.leverage_multiplier = leverage_multiplier

        self.trade_log = [] # STRUCTURE: [{ts:, symbol:, side:, quantity:, price:, trading_cost:, effective_price: }
        self.value_log = [] # STRUCTURE: [{ts:, net_equity:, equity:, cash:, buying_power: }
        self.positions = [] # STRUCTURE: [{

        self.cash = starting_cash
        self.buying_power = self.cash * leverage_multiplier
        self.backtest_ran = False


    def run_backtest(self):
        if self.backtest_ran:
            raise Exception("Backtest already run")
        self.backtest_ran = True

        # Iterate through each entry in signals and execute trades as needed
        for t in self.signals_df.index:
            pass


    def get_backtest_results(self):
        return self.trade_log, self.value_log