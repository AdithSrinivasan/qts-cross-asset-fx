"""
Stage 1: Hedging Out Macro and Risk Factors

Regresses each currency's daily excess return on carry, the dollar factor,
and sovereign CDS changes (EM only) to isolate the component of returns
orthogonal to systematic macro and risk premia.
"""

import os
import pandas as pd
from pathlib import Path

from src.data import load_fx_spot_pandas

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = pd.to_datetime("2022-01-03")
END_DATE = pd.to_datetime("2026-02-20")

# Load FX Data

if (DATA_DIR / "fx_data.csv").is_file():
    fx = pd.read_csv(DATA_DIR / "fx_data.csv", index_col="date", parse_dates=True)
else:
    fx = load_fx_spot_pandas(fx_data_dir=FX_DATA, start_date=START_DATE, end_date=END_DATE)

print("Time series of FX spot rates (Quote: USD)...")
print("Note: Currency per $1USD (values < 1 => currency is more valuable than USD)")
print(fx.head())

# Build Excess Return Data

