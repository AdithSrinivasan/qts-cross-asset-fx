import pandas as pd
from pandas import DataFrame
import logging
from typing import Optional, Dict, lifrom portfolio import Portfolio
from position import Position

class Backtester:

    def construct_threshold_dictionary(self, df: pd.DataFrame, is_train: bool) -> dict:
            """
            Constructs the Threshold Dictionary for entry/exit thresholds.
            Deals with the fact of whether we use Train/Test predictions

            Args:
                df (pd.DataFrame): threshold dictionary
                is_train (bool): tells us whether to use the 

            Returns:
                dict: dictionary of (key: Country (string), and value: float (threshold value))
            """
            if "Country" not in df.columns:
                raise ValueError("Countries to index by not in dataframe.")
            if is_train and  "Train Threshold" not in df.columns:
                raise ValueError("Train Threhshold column by not in dataframe.")
            if not is_train and  "Test Threshold" not in df.columns:
                raise ValueError("Test Threshold column not in dataframe.")
            
            # final dictionary we'll return
            d = {key: 0.0 for key in df["Country"]}
            
            # Reset index: without this Train/Test Threshold becomes _3 and _4 for names
            df.columns = ["Index", "Country", "Train_Threshold", "Test_Threshold"]
            
            # if is train, use Train Threshold
            if is_train:
                # drop 'Test Threshold' column
                for row in df.itertuples():
                    d[row.Country] = row.Train_Threshold
                # map from country to trai
            else:
                for row in df.itertuples():
                    d[row.Country] = row.Test_Threshold
            return d
            
    
    def __init__(self, 
                 return_predictions: DataFrame,
                 fx_futures: DataFrame,
                 entry_thresholds: dict,
                 exit_thresholds: dict,
                 contract_sizes: dict,
                 initial_margin_per_contract: dict,
                 is_train: bool,
                 contract_cost_fixed: float=0.07, # changed trading cost to 7 cents as discussed
                 starting_equity: float=2_000_000,
                 leverage_multiplier: float=5.0,
                 hedge_positions: bool=False):
        # initialize logging because it's certainly going to crash
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='app.log', # Saves to a file instead of the console
            # we want to append values, hence the 'a' over the 'w'.
            filemode='a'        
        )
        # load return predictions
        self.return_predictions = return_predictions
        self.fx_futures = fx_futures
        if "Country" not in entry_thresholds.columns:
            raise ValueError("Countries to index by not in dataframe.")
        
        # relevant countries
        self.countries =  [key for key in entry_thresholds["Country"]]
        
        # TODO: need a mapping from countries --> Futures?
        
        # construct the (k, v) dictionary where k = country (string), v = entry/exit threshold (float)
        self.entry_thresholds = self.construct_threshold_dictionary(entry_thresholds, is_train=is_train)
        self.exit_thresholds = self.construct_threshold_dictionary(exit_thresholds, is_train=is_train)
        self.contract_sizes = contract_sizes
        self.initial_margin_per_contract = initial_margin_per_contract
        self.leverage_multiplier = leverage_multiplier
        self.hedge_positions = hedge_positions

        self.trade_log = [] # STRUCTURE: [{ts:, symbol:, side:, quantity:, price:, trading_cost:, effective_price: }
        self.equity_log = [] # STRUCTURE: [{ts:, equity: }
        self.positions = [] # STRUCTURE: [{

        self.equity = starting_equity
        self.margin_used = 0
        self.buying_power = self.cash * leverage_multiplier
        self.backtest_ran = False

        
    
    # store positions?


    def run_backtest(self):
        if self.backtest_ran:
            raise Exception("Backtest already run")
        
        self.backtest_ran = True
        
        
        # return = net Pnl / cash
        
        # Portfolio class
        # for each position, instead of individual change

        # Iterate through each entry in signals and execute trades as needed
        for date in self.return_predictions.index:
            for country in self.return_predictions.columns:
                return_prediction = self.return_predictions.loc[date, country]
                entry_threshold = self.entry_thresholds[country]
                exit_threshold = self.exit_thresholds[country]

                fx_price = self.fx_futures.loc[date, country]
                
                """
                Every single position, when opened, has
                initial margin required.
                To open a new position, do we have free margin available,
                
                Free margin = equity - used margin
                
                Equity: previous days equity + pnl
                
                Log how much margin
                
                do 1 position per country
                
                Example:
                If I have 10k and the margin is 1k,
                I use 1k margin to get one contract of 100k.
                If price goes from 2 to 4, then
                our PL is 1 * 100k * (4-2) = 200k,
                new equity = 10k + 200k. 
                """
                margin_used = self.portfolio.get_margin_used() #TODO implement
                free_margin = self.equity - margin_used

                # Check if signal is buy
                if return_prediction >= entry_threshold:
                    num_countries = len(self.return_predictions.columns)

                    trade_value = self.equity / num_countries
                    contract_unit = self.contract_sizes[country]            # e.g., 125000
                    trade_qty = trade_value / (fx_price * contract_unit)       # contracts (float)

                    # TODO usually you must round to an integer contract count:
                    trade_qty = trade_qty  # floor, or round, depending on rules

                    trade_margin = trade_qty * self.initial_margin_per_contract[country]

                    # check free margin before placing:
                    if free_margin >= trade_margin and trade_qty > 0:
                        self.portfolio.update_position(country=country, side="LONG", trade_qty=trade_qty, notional=trade_value, initial_margin=trade_margin)                


                # Check if signal is a short 
                elif return_prediction <= -entry_threshold:
                    pass


                # For existing positions check if price is below exit band


                # Calculate net PL


            # Update equity 





    def get_backtest_results(self):
        return self.trade_log, self.equity_log