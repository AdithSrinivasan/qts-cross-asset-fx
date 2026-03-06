import pandas as pd
from linearmodels.panel import PanelOLS
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def run_ols(y, x, add_const=True, verbose=False):
    combined = pd.concat([y, x], axis=1).dropna()
    y_clean  = combined.iloc[:, 0]
    x_clean  = combined.iloc[:, 1:]
    X        = add_constant(x_clean) if add_const else x_clean
    model    = OLS(y_clean, X).fit()
    if verbose:
        print(model.summary())
    return model

def stage1_panel_regression(
    rx, carry, dollar, cds=None
):
    # --- long format ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")

    df = df.merge(
        dollar.rename("dollar"),
        left_on="date",
        right_index=True,
        how="inner"
    )

    if cds is not None:
        cds_long = cds.stack().rename("cds").reset_index()
        cds_long.columns = ["date", "currency", "cds"]
        df = df.merge(cds_long, on=["date", "currency"], how="inner")

    df = df.set_index(["currency", "date"]).sort_index()

    exog_vars = ["carry", "dollar"]
    if cds is not None:
        exog_vars.append("cds")

    exog = df[exog_vars]

    model = PanelOLS(
        df["rx"],
        exog,
        entity_effects=True
    )

    res = model.fit(
        cov_type="clustered",
        cluster_time=True
    )

    # Extract slope betas
    betas = res.params.copy()

    return res, betas


def stage1_panel_regression_cds(
    rx, carry, dollar, cds=None, base_ccy="AUD"
):
    # --- long format for rx & carry ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")

    # dollar (Series indexed by date)
    df = df.merge(dollar.rename("dollar"), left_on="date", right_index=True, how="inner")

    # optional CDS
    if cds is not None:
        cds_long = cds.stack().rename("cds").reset_index()
        cds_long.columns = ["date", "currency", "cds"]  # <<< fixes KeyError
        df = df.merge(cds_long, on=["date", "currency"], how="inner")  # inner = keep only where cds exists

    # panel index
    df = df.set_index(["currency", "date"]).sort_index()

    # --- currency-specific dollar betas: common dollar + (N-1) interactions ---
    currencies = df.index.get_level_values("currency")
    dummies = pd.get_dummies(currencies)

    if base_ccy not in dummies.columns:
        raise ValueError(f"base_ccy={base_ccy} not in currencies: {list(dummies.columns)}")

    for c in dummies.columns:
        if c == base_ccy:
            continue
        df[f"dollar_x_{c}"] = df["dollar"] * dummies[c].to_numpy()

    exog_vars = ["carry", "dollar"] + [f"dollar_x_{c}" for c in dummies.columns if c != base_ccy]
    if cds is not None:
        exog_vars.append("cds")

    exog = df[exog_vars]

    model = PanelOLS(df["rx"], exog, entity_effects=True)
    res = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    # recover dollar betas
    betas = {base_ccy: res.params["dollar"]}
    for c in dummies.columns:
        if c == base_ccy:
            continue
        betas[c] = res.params["dollar"] + res.params[f"dollar_x_{c}"]
    betas = pd.Series(betas).reindex(sorted(dummies.columns))

    return res, betas


