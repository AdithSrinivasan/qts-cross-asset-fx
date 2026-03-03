"""
Stage 1: Hedging Out Macro and Risk Factors

Regresses each currency's daily excess return on carry, the dollar factor,
and sovereign CDS changes (EM only) to isolate the component of returns
orthogonal to systematic macro and risk premia.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import prepare_fx_carry_data, calculate_fx_excess_returns
from src.regression import stage1_panel_regression, stage1_panel_regression_cds

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = pd.to_datetime("2022-01-03")
END_DATE = pd.to_datetime("2026-02-20")

# --- FX Excess Returns ---
fx_ret = calculate_fx_excess_returns(DATA_DIR, FX_DATA, START_DATE, END_DATE)
print("Time series of Excess Return on FX...")
print(fx_ret.head())

# --- Carry ---
carry_path = DATA_DIR / "carry.csv"
if carry_path.is_file():
    carry = pd.read_csv(carry_path, index_col="date", parse_dates=True)
else:
    # IMPORTANT: this must be your carry builder, not spot
    # e.g. carry = prepare_fx_carry_data(data_dir=data_dir, start_date=start_date, end_date=end_date)
    carry = prepare_fx_carry_data(
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE,
    )
print("Loaded daily FX carry data...")
print(carry.head())

# --- Dollar Factor ---
dollar = fx_ret.mean(axis=1)
dollar.name = "Dollar"

print("Constructed Dollar Factor (cross-sectional average FX excess return)...")
print(dollar.head())

# --- CDS ---
cds_path = DATA_DIR / "cds_5y_data.xlsx"
cds = pd.read_excel(cds_path, index_col="Dates", parse_dates=True)
cds = cds.reset_index()
cds = cds.rename(columns={"Dates": "date"})
cds = cds.rename(columns={
    "REPSOU CDS USD SR 5Y D14 Corp": "ZAR",
    "BRAZIL CDS USD SR 5Y D14 Corp": "BRL",
    "MEX CDS USD SR 5Y D14 Corp": "MXN",
    "JGB CDS USD SR 5Y D14 Corp": "JPY",
})
cds = cds.reindex(sorted(cds.columns), axis=1)
cds = cds.set_index("date")

cds = cds / 100 / 100 # convert from bps to decimal
print("CDS Data (in decimal)...")
print(cds.head())

# --- Regression Results ---

# --- currency sets ---
ccy_all = ["AUD", "CAD", "GBP", "JPY", "MXN", "ZAR"]
ccy_cds = ["JPY", "MXN", "ZAR"]

# =========================
# 1) MAIN: Stage 1 without CDS (full sample)
# =========================
res_main, betas_main = stage1_panel_regression_cds(
    rx=fx_ret[ccy_all],
    carry=carry[ccy_all],
    dollar=dollar,
    cds=None,
    base_ccy="AUD",
)

u_main = res_main.resids.unstack("currency")   # dates x currencies
u_main.index.name = "date"

# =========================
# 2) ROBUSTNESS: Stage 1 with CDS (CDS subset only)
# =========================
res_cds, betas_cds = stage1_panel_regression_cds(
    rx=fx_ret[ccy_cds],
    carry=carry[ccy_cds],
    dollar=dollar,
    cds=cds[ccy_cds],          # make sure cds is aligned + in decimal (bps/10000) or your z-score
    base_ccy="JPY",            # choose a base inside the subset
)

u_cds = res_cds.resids.unstack("currency")     # dates x currencies
u_cds.index.name = "date"

# =========================
# Optional: quick comparison outputs
# =========================
print("=== Stage 1 (No CDS) ===")
print(res_main.summary)
print("\nDollar betas (no CDS):\n", betas_main)

print("\n=== Stage 1 (With CDS subset) ===")
print(res_cds.summary)
print("\nDollar betas (with CDS):\n", betas_cds)

# --- Export Residuals to CSV for Stage 2 ---

print("\n=== Stage 1 (Without CDS subset) ===")
stage1_res_path = DATA_DIR / "stage1_residuals.csv"
res, betas = stage1_panel_regression(fx_ret, carry, dollar, cds=None)
u_hat_wide = res.resids.unstack("currency")
u_hat_wide.index.name = "date"
u_hat_wide.to_csv(stage1_res_path)
print("Residuals for Stage 2 Regression exported to...", stage1_res_path)

# IMPORTANT FOR STAGE 2 REGRESSION: MAKE SURE TO ALIGN DATES WITH EQUITY PREDICTORS
# u2, X2 = u_hat_wide.align(X, join="inner", axis=0)   # align on dates
# # if X also has same currency columns:
# u2, X2 = u2.align(X2, join="inner", axis=1)         # align on currencies