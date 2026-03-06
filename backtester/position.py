
from collections import defaultdict
from typing import Any

class Position:
    """
    
    The Position Class
    
    Overall Summary
    Tracks the aggregate Position for a country for all trades that occur during
    backtesting.
    """
    
    def __init__(self, country):
        """
        
        Initializes our Position Class

        Args:
            country (str): the country we're working with
        
        Attributes & their justifications:

        """        
        self.country = country
        # safe to make it None than to give it an actual value
        self.last_price = None
        self.prev_prices = {}
        self.notional = 0.0
        self.net_qty = 0.0
        self.margin = 0.0
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
    
    def update(self, trade_qty, notional, initial_margin):
        """
        _summary_

        Args:
            trade_qty (_type_): _description_
            notional (_type_): _description_
            initial_margin (_type_): _description_
        """
        self.net_qty += trade_qty
        
        # we assume notional is not signed and default positive, so if we're selling, change notional to negative
        if trade_qty < 0:
            notional *= -1
        self.notional += notional
        self.margin += initial_margin

    def get_notional(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        return self.notional
    
    def get_quantity(self) -> float:
        """
        
        Getter for self.net_qty

        Returns:
            float: _description_
        """    
        return self.net_qty

    # Save last price and return PL
    def calc_today_pnl(self, date, new_price: int) -> float:
        """_summary_

        Args:
            date (_type_): _description_
            new_price (int): _description_

        Returns:
            float: _description_
        """ 
        if date in self.prev_prices:
            return 0.0
        if not self.last_price:
            self.prev_prices[date] = new_price
            self.last_price = new_price
            return 0.0
        
        # we calculate pnl by this formula: net quantity
        pnl = self.net_qty * self.notional * (new_price - self.last_price)
        # add date to previous prices and update last price to be the new_price given in the function
        self.prev_prices[date] = new_price
        self.last_price = new_price
        return pnl

    # Update position (buy/sell)
    def get_country(self):
        return self.country
    