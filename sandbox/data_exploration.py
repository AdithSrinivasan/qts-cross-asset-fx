import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

df = pd.read_excel(DATA_DIR / "equity_indices.xlsx", parse_dates=["Dates"])
df.set_index("Dates", inplace=True)
df = df.dropna()
print(df.head())

df = df.pct_change()
print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))