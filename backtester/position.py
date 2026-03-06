


class CountryPosition:
    
    def __init__(self, country, total_notional, total_margin_used, maintenance_margin_needed=0.0):
        self.country = country
        self.pnl = 0.0
        self.total_notional = total_notional
        # we add notional (effective exposure)
        # Total notional
        # notional not the same
        # Total margin used
        self.total_margin_used = total_margin_used
        # Maintence margin needed (TODO imlement in future)
        self.maintenance_margin_needed = maintenance_margin_needed

    def add_contract(self):
        pass


    # Save last price and return PL
    def calc_pnl(self, new_price: int) -> float:
        pass

    # Update position (buy/sell)
    def get_country(self):
        return self.country