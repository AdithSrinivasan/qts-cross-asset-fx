import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import percentileofscore

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = "2022-01-03"
END_DATE = "2026-01-01"

def empirical_pvalue(data, value, tail='two'):
    """
    Parameters
    ----------
    data  : array of empirical distribution
    value : observed value to test
    tail  : 'two', 'left', 'right'
    """
    # percentileofscore gives percentile (0-100), convert to (0-1)
    percentile = percentileofscore(data, value) / 100

    if tail == 'right':
        return 1 - percentile
    elif tail == 'left':
        return percentile
    elif tail == 'two':
        return 2 * min(percentile, 1 - percentile)

data = pd.read_csv(DATA_DIR / "rf_train_predictions.csv")
data = data[data.date >= START_DATE]
data = data[data.date <= END_DATE]
data.set_index("date", inplace=True)


preds = pd.read_csv(DATA_DIR / "rf_signals.csv")
preds = preds[preds.date >= START_DATE]
preds = preds[preds.date <= END_DATE]
preds.set_index("date", inplace=True)

p_values = pd.DataFrame(index=preds.index, columns=data.columns)

for col in data.columns:
    empirical_dist = data[col].dropna().values
    p_values[col] = [
        empirical_pvalue(empirical_dist, v, tail='two')
        for v in preds[col]
    ]

print(p_values.head())
p_values.to_csv(DATA_DIR / "rf_signals_pvalues.csv")
