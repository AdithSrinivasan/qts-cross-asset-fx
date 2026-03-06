import pandas as pd
from pandas import DataFrame
import logging
from typing import Optional, Dict, Any
from portfolio import Portfolio
from position import Position

class Backtester:
    def __init__(self, 
                 return_predictions: DataFrame,
                 fx_futures_panel: DataFrame,
                 entry_thresholds: pd.DataFrame,
                 exit_thresholds: pd.DataFrame,
                 fx_contract_specs: pd.DataFrame,
                 is_train: bool,
                 contract_cost_fixed: float=0.07, # changed trading cost to 7 cents as discussed
                 starting_equity: float=2_000_000,
                 leverage_multiplier: float=5.0,
                 position_weight_ratio: float=2,
                 hedge_positions: bool=False,
                 hedge_ratio=None):

        # load return predictions
        self.return_predictions = return_predictions.copy()
        self.fx_futures_panel = fx_futures_panel.copy()
        if "Country" not in entry_thresholds.columns:
            raise ValueError("Countries to index by not in dataframe.")
        
        # relevant countries
        self.countries =  [key for key in entry_thresholds["Country"]]
                
        # construct the (k, v) dictionary where k = country (string), v = entry/exit threshold (float)
        self.entry_thresholds = construct_threshold_dictionary(entry_thresholds, is_train=is_train)
        self.exit_thresholds = construct_threshold_dictionary(exit_thresholds, is_train=is_train)
        print(self.entry_thresholds)
        print(self.exit_thresholds)

        # Gets fx contract specs
        self.contract_sizes, self.initial_margin_per_contract = parse_contract_specs(fx_contract_specs)

        # Sets leverage and hedging values
        self.contract_cost_fixed = contract_cost_fixed
        self.leverage_multiplier = leverage_multiplier
        self.position_weight_ratio = position_weight_ratio
        self.hedge_positions = hedge_positions
        self.hedge_ratio = hedge_ratio
        
        # create self.portfolio object
        self.portfolio = Portfolio()

        self.trade_log = [] # STRUCTURE: [{ts:, symbol:, side:, quantity:, price:, trading_cost:, effective_price: }
        self.portfolio_log = [] # STRUCTURE: [{ts:, equity: }
        self.positions = [] # STRUCTURE: [{

        self.equity = starting_equity
        self.margin_used = 0
        self.backtest_ran = False


    def run_backtest(self):
        if self.backtest_ran:
            raise Exception("Backtest already run")
        self.backtest_ran = True

        # Iterate through each entry in signals and execute trades as needed
        for date in self.return_predictions.index:
            day_pl = 0
            for country in self.return_predictions.columns:
                return_prediction = self.return_predictions.loc[date, country]
                entry_threshold = self.entry_thresholds[country]
                exit_threshold = self.exit_thresholds[country]
                
                # Check that we have fx price for this date 
                if (
                    date not in self.fx_futures_panel.index
                    or country not in self.fx_futures_panel.columns
                    or pd.isna(self.fx_futures_panel.loc[date, country])
                ):
                    print(f"Skipping {date} {country} because FX data missing")
                    continue
                fx_price = self.fx_futures_panel.loc[date, country]

                margin_used = self.portfolio.get_margin_used()
                free_margin = self.equity - margin_used

                # Check if signal is long or short
                if return_prediction >= entry_threshold or return_prediction <= -entry_threshold:
                    num_assets = len(self.return_predictions.columns)
                    contract_multiplier = self.contract_sizes[country]

                    target_asset_exposure = self.equity * self.leverage_multiplier * self.position_weight_ratio / num_assets
                    current_asset_exposure = self.portfolio.get_current_exposure()
                    contract_value = fx_price * contract_multiplier
                    trade_qty = (target_asset_exposure - current_asset_exposure) / (contract_value + self.contract_cost_fixed)
                    trade_margin = trade_qty * self.initial_margin_per_contract[country]

                    # check free margin before placing:
                    if free_margin >= trade_margin and abs(trade_qty) > 0:
                        # Add position with correct side
                        trade_sign = 1 if return_prediction >= entry_threshold else -1
                        trade_qty *= trade_sign
                        trade_value = trade_qty * fx_price * contract_multiplier
                        self.portfolio.update_position(country=country, trade_qty=trade_qty, notional=trade_value, initial_margin=trade_margin)
                        self.trade_log.append({"date": date, "country": country, "qty": trade_qty, "trade_price": fx_price, "trade_value": trade_value, "trade_margin": trade_margin})


                # For existing positions check if price is below exit band
                cur_position = self.portfolio.get_position(country=country)
                if cur_position:
                    cur_qty = cur_position.get_quantity()
                    if (cur_qty > 0 and return_prediction <= exit_threshold) or (cur_qty < 0 and return_prediction >= -exit_threshold): 
                        self.portfolio.liquidate_position(country=country)


                # Calculate new net PL
                new_pl = self.portfolio.get_today_pnl(country=country, date=date, new_p=fx_price)
                day_pl += new_pl

            # Update equity 
            self.equity += day_pl
            
            # add to portfolio log
            target_asset_exposure = self.equity * self.leverage_multiplier * self.position_weight_ratio / len(self.return_predictions.columns)
            self.portfolio_log.append({"date": date, "equity": self.equity, "pl": day_pl, "margin_used": self.portfolio.get_margin_used(), "total_exposure": self.portfolio.get_current_exposure(), "target_asset_exposure": target_asset_exposure})


    def get_backtest_results(self):
        return self.trade_log, self.portfolio_log
    


def construct_threshold_dictionary(df: pd.DataFrame, is_train: bool) -> dict:
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
            for index, row in df.iterrows():
                d[row["Country"]] = row["Train_Threshold"]
            # map from country to trai
        else:
            for index, row in df.iterrows():
                d[row["Country"]] = row["Test_Threshold"]
        return d


def parse_contract_specs(fx_contract_specs: pd.DataFrame):
    contract_sizes = {}
    initial_margin_per_contract = {}

    # Iterate through each nation saving contract specs
    for index, row in fx_contract_specs.iterrows():
        contract_sizes[row["Country"]] = row["Contract Size"]
        initial_margin_per_contract[row["Country"]] = row["Initial Margin"]

    return contract_sizes, initial_margin_per_contract
