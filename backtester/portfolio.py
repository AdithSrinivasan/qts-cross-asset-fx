from position import Position


class Portfolio:
    
    def __init__(self, cash):
        self.positions = {}
        self.cash = cash
        self.trades = []
    
    def update_position(self, country, side, notional, initial_margin, trade_qty):
        if country not in self.positions:
            pass
        else:
            pass
    
    def remove_position():
        del 
        pass
    
    def liquidate():
        pass
    
    def update_position(position):
        pass
    
    def get_margin_used(self):
        pass
    
    def calc_net_pnl(self) -> float:
        """
        
        Calculates net P&L for the overall Portfolio.
        Implicitly, this calls 

        Returns:
            float: _description_
        """
    
