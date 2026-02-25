# qts-cross-asset-fx

Winter 2026 FINM 33150 Quantitative Trading Strategies — Cross-Asset FX Futures Trading Project

Research project for the FINM 33150 Quantitative Trading Strategies course, in fulfilment of the MS Financial Mathematics at the University of Chicago. The objective is to study whether lead–lag relationships between **equity and credit markets** can predict **FX futures returns**, and to evaluate tradable strategies under realistic trading assumptions.

## Researchers
- Adith Srinivasan
- Andrew Moukabary
- Cole Koryto: 12506473
- Jonathan Kim
- Scott Hanna

## Project Overview
This repository contains a research and backtesting framework for testing cross-asset predictability using equity (and credit proxies where relevant) as signals for FX futures. The strategy focuses on a small set of developed and emerging markets with varying liquidity, using standardized FX futures for implementation. Evaluation is conducted via walk-forward backtesting with transaction costs and contract rolls.

**Core questions**
- Do equity/credit signals lead FX futures in select markets?
- Are effects stronger in less liquid or higher FX-demand markets?
- Are signals robust out-of-sample and across market regimes?

## Universe (Prototype)
Initial focus on ~5–6 countries selected from Bloomberg DM/EM indexes (diverse regions; mix of liquidity levels). Final selection subject to data availability.

## Methodology
- **Signals (X):** equity index returns, factor returns, volatility/regime features; optional credit proxies (e.g., CDS where available).
- **Targets (Y):** next-period FX futures returns (excess returns where appropriate).
- **Models:** linear baselines (ridge/lasso), with optional tree-based models.
- **Portfolio:** cross-sectional ranking on FX futures; risk controls; transaction costs; futures rolls.
- **Validation:** rolling walk-forward backtests; IC by year; regime analysis.

## Data & Tools
- **Data:** Bloomberg (DM/EM indexes, FX futures), DataVento (as needed), selected public macro series.
- **Stack:** Python, Jupyter, pandas/numpy, statsmodels/sklearn; plotnine/matplotlib for visualization.

## Timeline
- **Draft due:** Friday night (for Saturday 9am submission)

## Notes / Risks
- Strict timing enforced to avoid lookahead (equity information at _t_ mapped to FX returns at _t+1_).
- FX liquidity differences expected to attenuate signal strength in the most liquid pairs.
- Overfitting controlled via walk-forward validation and simple baselines prior to complex models.
