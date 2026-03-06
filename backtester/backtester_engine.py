import numpy as np
import pandas as pd
from pandas import DataFrame
from portfolio import Portfolio
from hedge import get_equity_returns, get_hedge_returns

class Backtester:
    def __init__(self, 
                 return_predictions: DataFrame,
                 fx_futures_panel: DataFrame,
                 entry_thresholds: pd.DataFrame,
                 exit_thresholds: pd.DataFrame,
                 fx_contract_specs: pd.DataFrame,
                 is_train: bool,
                 contract_cost_fixed: float=1.7, # changed trading cost to 7 cents as discussed
                 starting_equity: float=2_000_000,
                 leverage_multiplier: float=5.0,
                 position_weight_ratio: float=2,
                 hedge_positions: bool=False,
                 hedge_ratio=None):

        # load return predictions
        self.return_predictions = return_predictions.copy()
        self.fx_futures_panel = fx_futures_panel.copy()
        if "Country" not in entry_thresholds.columns: # TODO add more checks
            raise ValueError("Countries to index by not in dataframe.")

        # Gets the number of country assets we are running on TODO should be min of all dfs
        self.num_assets = self.fx_futures_panel.notna().any().sum()

        # construct the (k, v) dictionary where k = country (string), v = entry/exit threshold (float)
        self.entry_thresholds = construct_threshold_dictionary(entry_thresholds, is_train=is_train)
        self.exit_thresholds = construct_threshold_dictionary(exit_thresholds, is_train=is_train)

        # Gets fx contract specs
        self.contract_sizes, self.initial_margin_per_contract, self.maintenance_margin_per_contract = parse_contract_specs(fx_contract_specs)

        # Sets leverage and hedging values
        self.contract_cost_fixed = contract_cost_fixed
        self.leverage_multiplier = leverage_multiplier
        self.position_weight_ratio = position_weight_ratio
        self.hedge_positions = hedge_positions
        self.hedge_ratio = hedge_ratio
        
        # create self.portfolio object
        self.portfolio = Portfolio()

        self.trade_log = [] # STRUCTURE: 
        self.portfolio_log = [] # STRUCTURE: 

        self.equity = starting_equity
        self.backtest_ran = False
        self.total_trading_fees = 0

        self.hedge_returns = None


    def run_backtest(self):
        if self.backtest_ran:
            raise Exception("Backtest already run.")
        self.backtest_ran = True

        # Iterate through each entry in signals and execute trades as needed
        for date in self.return_predictions.index:
            day_pl = 0
            day_country_pnl = {}
            for country in self.return_predictions.columns:
                return_prediction = self.return_predictions.loc[date, country]
                entry_threshold = self.entry_thresholds[country]
                exit_threshold = self.exit_thresholds[country]
                
                # Check that we have fx price for this date and update portfolio price for asset
                if (
                    date not in self.fx_futures_panel.index
                    or country not in self.fx_futures_panel.columns
                    or pd.isna(self.fx_futures_panel.loc[date, country])
                ):
                    print(f"Skipping {date} {country} because FX data missing")
                    continue
                fx_price = self.fx_futures_panel.loc[date, country]
                self.portfolio.update_asset_price(country=country, new_price=fx_price, date=date)

                margin_used = self.portfolio.get_margin_used()
                free_margin = self.equity - margin_used

                # Check if signal is long or short
                if return_prediction >= entry_threshold or return_prediction <= -entry_threshold:
                    contract_multiplier = self.contract_sizes[country]

                    target_asset_exposure = self.equity * self.leverage_multiplier * self.position_weight_ratio / self.num_assets
                    current_asset_exposure = self.portfolio.get_current_country_exposure(country=country)
                    contract_value = fx_price * contract_multiplier
                    trade_qty = (target_asset_exposure - current_asset_exposure) / (contract_value + self.contract_cost_fixed)
                    trade_margin = trade_qty * self.initial_margin_per_contract[country]

                    # check free margin before placing:
                    if free_margin >= trade_margin and abs(trade_qty) > 0:
                        # Add position with correct side
                        trade_sign = 1 if return_prediction >= entry_threshold else -1
                        trade_qty *= trade_sign
                        trade_value = trade_qty * fx_price * contract_multiplier

                        # Account for fixed trading costs
                        trading_cost = abs(trade_qty * self.contract_cost_fixed)
                        self.equity -= trading_cost
                        self.total_trading_fees += trading_cost

                        # Log trade
                        self.portfolio.update_position(country=country, price=fx_price, trade_qty=trade_qty, contract_multiplier=contract_multiplier, contract_initial_margin=self.initial_margin_per_contract[country], contract_maintenance_margin=self.maintenance_margin_per_contract[country])
                        self.trade_log.append({"date": date, "country": country, "qty": trade_qty, "trade_price": fx_price, "trade_value": trade_value, "trade_margin": trade_margin, "trading_cost": trading_cost})

                # For existing positions check if price is below exit band
                cur_position = self.portfolio.get_position(country=country)
                if cur_position:
                    cur_qty = cur_position.get_quantity()
                    if (cur_qty > 0 and return_prediction <= exit_threshold) or (cur_qty < 0 and return_prediction >= -exit_threshold):
                        self.portfolio.liquidate_position(country=country)

                # Calculate new net PL TODO should this be earlier?
                new_pl = self.portfolio.get_today_pnl(country=country)
                day_pl += new_pl
                day_country_pnl[country] = new_pl

            # Check if margin call and liquidate position with the highest relative pl for day
            total_maintenance_margin = self.portfolio.get_maintenance_margin_used()
            free_margin = self.equity - self.portfolio.get_margin_used()
            if free_margin < total_maintenance_margin:
                min_rel_pl = float("inf")
                loss_country = None
                for country in self.portfolio.get_open_asset_names():
                    rel_pl = self.portfolio.get_today_pnl(country) / self.portfolio.get_position(country=country).get_exposure() # TODO refactor
                    if rel_pl < min_rel_pl:
                        min_rel_pl = rel_pl
                        loss_country = country
                print(f"MARGIN CALL: LIQUIDATING {loss_country}")
                self.portfolio.liquidate_position(country=loss_country)

            # TODO call hedging function to update hedge (be sure to include in PL)


                # Calculate new net PL
                new_pl = self.portfolio.get_today_pnl(country=country, date=date, new_p=fx_price)
                day_pl += new_pl

            # Update equity
            self.equity += day_pl
            
            # TODO call hedging function to update hedge (be sure to include in PL)
            if self.hedge_positions:
                # get the dollar return at that date
                dollar_return = self.get_dollar_return(date=date)

                # hedge pl is our ratio multiplied by our exposure and our dollar return
                hedge_pl = self.hedge_ratio * self.portfolio.get_total_exposure() * dollar_return
            else:
                hedge_pl = 0.0
            day_pl += hedge_pl

            # add to portfolio log
            target_total_exposure = self.equity * self.leverage_multiplier
            self.portfolio_log.append({"date": date, "equity": self.equity, "pl": day_pl, "margin_used": self.portfolio.get_margin_used(), "total_maintenance_margin": self.portfolio.get_maintenance_margin_used(), "total_exposure": self.portfolio.get_total_exposure(), "target_total_exposure": target_total_exposure, "total_trading_fees": self.total_trading_fees, "num_positions": self.portfolio.get_num_positions(), "country_pnl": day_country_pnl})


    def get_dollar_return(self, date):
        if self.hedge_returns is None:
            hedge_returns = get_hedge_returns()
            hedge_returns.index = pd.to_datetime(hedge_returns.index)
            self._hedge_returns = hedge_returns["hedge_returns"].to_dict()
        return float(self.hedge_returns.get(pd.to_datetime(date), 0.0))


    def get_backtest_results(self):
        if not self.backtest_ran:
            raise Exception("Backtest not yet ran.")
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
        # print(df.columns)
        if "Country" not in df.columns:
            raise ValueError("Countries to index by not in dataframe.")
        if is_train and  not ("Train Threshold" in df.columns or "Train_Threshold" in df.columns):
            raise ValueError("Train Threhshold column by not in dataframe.")
        if not is_train and  not ("Test Threshold" in df.columns or "Test_Threshold" in df.columns):
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
    maintenance_margin_per_contract = {}

    # Iterate through each nation saving contract specs
    for index, row in fx_contract_specs.iterrows():
        contract_sizes[row["Country"]] = row["Contract Size"]
        initial_margin_per_contract[row["Country"]] = row["Initial Margin"]
        maintenance_margin_per_contract[row["Country"]] = row["Maintenance Margin"]

    return contract_sizes, initial_margin_per_contract, maintenance_margin_per_contract
