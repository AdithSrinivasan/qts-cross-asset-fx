from position import Position


class Portfolio:
    
    def __init__(self):
        """
        Initializer for Portfolio Class.
        
        Inputs:
            self.positions: dictionary that maps each country to its Position object
        """
        self.positions = {}
    
    def update_position(self, country, trade_qty: float, notional, initial_margin):
        """
        If we make a new trade for a country, update it accordingly by calling 
        position.update(trade quantity, notional value, initial margin).
        That handles most of the functionality
        
        The trade_qty also gives us the idea of if we should buy or sell based on its sign.
        Ex: +1.0 buy 1 contract, -2.6 sell 2.6 contracts.
        We assume trade quantity is a float and can be fractional.
        
        But this idea is handled more rigorously in the country's Position object. In
        particular, the update() function called below.
        
        Inputs: 
            country (str): country in self.positions that we need to call, if it exists.
            trade_qty (float): the quantity/number of contracts we need to trade. positive means buy,
                negative means sell.
            notional (float): The total theoretical value of the contracts we have (price * quantity)
            initial margin (float): The minimum amount of equity we need to properly open a futures
                position.
            
        """
        # make sure we're instantiating the value to self.positions
        if country not in self.positions:
            # create new position object in self.positions, with key to each country
            self.positions[country] = Position(country)
        # don't add updates with no trade quantity
        if trade_qty == 0.0:
            return
        # get pointer to current aggregate position for the country that
        position = self.positions[country]
        # call position.update() to update notional position. this handles the main functionality
        position.update(trade_qty, notional, initial_margin)
        
    
    def get_margin_used(self) -> float:
        """
        Aggregates the used margin for all positions in all countries
        
        Inputs: 
            None, besides the Portfolio object we have
        
        Output: 
            float telling us the total margin used in the portfolio
        """
        total_margin = 0.0
        # compute the notional for each position. don't need country, just need notional.
        for _, position in self.positions.items():
            # get that notional value for the position. Pre-caclulated and updated during update() calls in update_position()
            total_margin += position.get_margin()
        # return our portfolio's total margin
        return total_margin
    
    def get_position(self, country):
        """
        Gets the Position object for a country 
        during backtesting. Also helpful for debugging/examining
        a particular country
        
        Inputs:
            country (str): country
        
        Outputs:
            position (Position): Position
        """
        # check if the country exists for proper pnl usage
        if country not in self.positions:
            return None
        
        # get the position via the k, v pair in the dictionary to access the position desired
        return self.positions[country]

    def liquidate_position(self, country) -> None:
        """
        Given a country, completely eliminate it
        marginal goes to 0, notional to 0
        
        Input: 
            Country (str): country in self.positions that we need to call, if it exists.
        
        Outputs: 
            Nothing, but liquidates position/removes country and Position object from self.positions
        """
        # check if the country exists for proper pnl usage
        if country not in self.positions:
            return None
        
        # # get the position via the k, v pair in the dictionary
        # self.positions[country].liquidate()
        
        # delete country from self.positions in our Portfolio object
        del self.positions[country]
        
        # leave function body
        return None
    
    def get_total_exposure(self) -> float:
        """
        Tells us current exposure.
        
        Formula: sum of quantity we have * contract multiplier * current asset price <- gives total exposure
        
        
        for every product
        
        divided by all of the products in the portfolio

        Returns:
            float: _description_
        """      
        exposure = 0.0
        for _, position in self.positions.items():
            exposure += position.get_notional()
        return exposure
    
    def get_current_country_exposure(self, country):
        if country not in self.positions:
            return 0.0
        position = self.positions[country]
        return position.get_notional()
    
    def get_num_positions(self) -> int:
        return len(self.positions)

    def get_today_pnl(self, country, date, new_p) -> float:
        """
        country date and price
        Calculates net P&L for the overall Portfolio.
        Implicitly, this calls calc_today_pnl in the country's Position object.
        
        Inputs:
            country (str): tells us which country
            date (str): used in position.prev_prices to 
            ensure the associated price on that date didn't exist beforehand.

        Returns:
            float: today's profit and loss for the existing country
        """
        
        # check if the country exists for proper pnl usage
        if country not in self.positions:
            return 0.0

        # if so, calculate today's profit and loss!
        return self.positions[country].calc_today_pnl(date, new_p)
        
    
