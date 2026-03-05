import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

sig = pd.read_csv("/Users/adithsrinivasan/Documents/GitHub/qts-cross-asset-fx/data/rf_signals.csv")

data = sig["Australia"].dropna().values

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram
ax.hist(data, density=True, alpha=0.4, color='steelblue', label='Empirical')

# Normal overlay
mu, std = data.mean(), data.std()
x = np.linspace(data.min(), data.max(), 200)
ax.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')

# KDE overlay
kde = gaussian_kde(data)
ax.plot(x, kde(x), 'g-', linewidth=2, label='KDE')

ax.set_title('Mexico Signal Distribution')
ax.set_xlabel('Signal')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()