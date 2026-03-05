import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import prepare_fx_carry_data, calculate_fx_excess_returns
from src.regression import stage1_panel_regression, stage1_panel_regression_cds, run_ols

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = "2022-01-03"
END_DATE = "2024-12-31"

# --- FX Excess Returns ---
fx_ret = calculate_fx_excess_returns(DATA_DIR, FX_DATA, START_DATE, END_DATE)
print("Time series of Excess Return on FX...")
print(fx_ret.tail())

# --- Dollar ETF ---

etf = pd.read_excel(DATA_DIR / "dollar_etf.xlsx", parse_dates=["Dates"])
etf.set_index("Dates", inplace=True)
etf_ret = etf.pct_change()
etf_ret.dropna(inplace=True)

print(etf_ret.head())

# --- Cut to In-Sample ---
fx_ret = fx_ret.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE)]
etf = etf.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE)]

# --- Collect In-Sample Country-Specific Hedge Ratios ---

results = []
for col in fx_ret.columns:
    model = run_ols(etf_ret, fx_ret[col], add_const=False, verbose=False)
    results.append({
        'fx'     : col,
        'beta'   : model.params.iloc[0],
        'std_err': model.bse.iloc[0],
        't_stat' : model.tvalues.iloc[0],
        'p_value': model.pvalues.iloc[0],
        'r2'     : model.rsquared
    })

beta_table = pd.DataFrame(results).set_index('fx')
beta_table.to_csv(DATA_DIR / "is_hedge_ratios.csv")
print(beta_table.round(4))