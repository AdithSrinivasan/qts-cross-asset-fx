import pandas as pd
from pandas import DataFrame
import logging
from typing import Optional, float, int

class Backtester:
    def __init__(self, 
                 # new dataframes generated from regression passed
                 train_predictions: Optional[DataFrame], 
                 test_predictions: Optional[DataFrame], 
                 entry_thresholds: Optional[DataFrame],
                 exit_thresholds: Optional[DataFrame],
                 # changed trading cost to 7 cents as discussed
                 trading_cost: float=0.07,
                 starting_cash: float=100_000.0,
                 q_trading_thres: float=0.2,
                 fx_look_ahead: int=1,
                 # maybe include max assets=5?
                 leverage_multiplier: float=5.0):
        
        # load 
        self.train_pred = train_predictions
        self.test_pred = test_predictions
        self.entry_thresh = entry_thresholds
        self.exit_thresh = exit_thresholds
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
        
        # initialize logging because it's certainly going to crash
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='app.log', # Saves to a file instead of the console
            
            # we want to append values
            filemode='a'        
        )
        self.backtest_ran = True

        # Iterate through each entry in signals and execute trades as needed
        


    def get_backtest_results(self):
        return self.trade_log, self.value_log