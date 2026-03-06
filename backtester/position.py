
from collections import defaultdict
from typing import Any

class Position:
    """
    
    The Position Class
    
    Overall Summary
    Tracks the aggregate Position for a country for all trades that occur during
    backtesting.
    """
    
    def __init__(self, country, price, contract_multiplier, contract_initial_margin, contract_maintenance_margin):
        """
        
        Initializes our Position Class
        Args:
            country (str): the country we're working with
        
        Attributes & their justifications:

        """        
        self.country = country
        self.contract_multiplier = contract_multiplier
        self.contract_initial_margin = contract_initial_margin
        self.contract_maintenance_margin = contract_maintenance_margin
        # safe to make it None than to give it an actual value
        self.last_price = price
        self.prices = {}
        self.exposure = 0.0
        self.net_qty = 0.0
        self.margin_used = 0.0
        self.maintenance_margin = 0.0
        # Maintence margin needed (TODO imlement in future)
        
        self.maintenance_margin_needed = 0.0
        """
        Notes: TODO: delete
        # dictionary of dates: float
        =# we add notional (effective exposure)
        # Total notional
        # notional not the same
        # Total margin used
        """

    def update_price(self, new_price, date):
        if date in self.prices:
            print("WARNING: OVERWRITING PRICE FOR EXISTING DAY")

        # add date to previous prices and update last price to be the new_price given in the function
        self.prices[date] = new_price
        self.last_price = new_price
        self.exposure = abs(self.net_qty) * self.last_price * self.contract_multiplier

    def update_position(self, trade_qty):
        """
        _summary_

        Args:
            trade_qty (_type_): _description_
        """
        self.net_qty += trade_qty
        self.exposure = abs(self.net_qty) * self.last_price * self.contract_multiplier
        self.margin_used = abs(self.net_qty) * self.contract_initial_margin
        self.maintenance_margin = abs(self.net_qty) * self.contract_maintenance_margin

    def get_exposure(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        return self.exposure
    
    def get_directional_exposure(self):
        if self.net_qty < 0.0:
            return -self.exposure
        return self.exposure
    
    def get_margin_used(self):
        return self.margin_used

    def get_maintenance_margin(self):
        return self.maintenance_margin
    
    def get_quantity(self) -> float:
        """
        
        Getter for self.net_qty

        Returns:
            float: _description_
        """    
        return self.net_qty

    # Save return PL
    def calc_pnl(self) -> float:
        """_summary_

        Args:

        Returns:
            float: _description_
        """
        if len(self.prices.values()) >= 2:
            yesterday_price = list(self.prices.values())[-2]
        else:
            return 0.0
        pnl = self.net_qty * (self.last_price - yesterday_price) * self.contract_multiplier
        return pnl

    # Update position (buy/sell)
    def get_country(self):
        return self.country
