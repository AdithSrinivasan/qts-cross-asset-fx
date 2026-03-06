import pandas as pd
from pathlib import Path
from linearmodels.panel import PanelOLS
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from src.load_data import prepare_fx_carry_data, calculate_fx_excess_returns


def build_regression_inputs():
    start_date = "2022-01-03"
    end_date = "2026-01-01"

    data_dir = Path(__file__).parent.parent / "data"
    fx_data = data_dir / "fx_data"

    # --- FX Excess Returns ---
    fx_ret = calculate_fx_excess_returns(data_dir, fx_data, start_date, end_date)
    print("Loaded time series of Excess Return on FX...")

    # --- Carry ---
    carry_path = data_dir / "carry.csv"
    if carry_path.is_file():
        carry = pd.read_csv(carry_path, index_col="date", parse_dates=True)
    else:
        # IMPORTANT: this must be your carry builder, not spot
        # e.g. carry = prepare_fx_carry_data(data_dir=data_dir, start_date=start_date, end_date=end_date)
        carry = prepare_fx_carry_data(
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
        )
    print("Loaded daily FX carry data...")

    # --- Dollar Factor ---
    dollar = fx_ret.mean(axis=1)
    dollar.name = "Dollar"

    print("Constructed Dollar Factor (cross-sectional average FX excess return)...")

    # --- CDS ---
    cds_path = data_dir / "cds_5y_data.xlsx"
    cds = pd.read_excel(cds_path, index_col="Dates", parse_dates=True)
    cds = cds.reset_index()
    cds = cds.rename(columns={"Dates": "date"})
    cds = cds.rename(
        columns={
            "REPSOU CDS USD SR 5Y D14 Corp": "ZAR",
            "BRAZIL CDS USD SR 5Y D14 Corp": "BRL",
            "MEX CDS USD SR 5Y D14 Corp": "MXN",
            "JGB CDS USD SR 5Y D14 Corp": "JPY",
            "KOREA CDS USD SR 5Y D14 Corp": "KRW",
            "SINGP CDS USD SR 5Y D14 Corp": "SGD",
            "HONGK CDS USD SR 5Y D14 Corp": "HKD",
            "INDIA CDS USD SR 1Y D14 Corp": "INR",
        }
    )

    cds = cds.reindex(sorted(cds.columns), axis=1)
    cds = cds.set_index("date")

    cds = cds / 100 / 100  # convert from bps to decimal
    print("Loaded CDS Data (in decimal)...")

    print("Completed construction of Stage 1 regression inputs!")

    return fx_ret, carry, dollar, cds


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
    rx, carry, dollar
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

    df = df.set_index(["currency", "date"]).sort_index()

    exog_vars = ["carry", "dollar"]

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

    return res


def stage1_panel_regression_cds(rx, carry, dollar, cds):
    # --- long format ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")
    df = df.merge(dollar.rename("dollar"), left_on="date", right_index=True, how="inner")

    if cds is not None:
        cds_long = cds.stack().rename("cds").reset_index()
        cds_long.columns = ["date", "currency", "cds"]
        df = df.merge(cds_long, on=["date", "currency"], how="inner")

    df = df.set_index(["currency", "date"]).sort_index()

    exog_vars = ["carry", "dollar"]
    if cds is not None:
        exog_vars.append("cds")

    model = PanelOLS(df["rx"], df[exog_vars], entity_effects=True)
    res = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    return res


def stage1_panel_regression_interactions(rx, carry, dollar, base_ccy="CAD"):
    # --- long format ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")
    df = df.merge(dollar.rename("dollar"), left_on="date", right_index=True, how="inner")
    df = df.set_index(["currency", "date"]).sort_index()

    # --- currency-specific dollar interaction terms ---
    currencies = df.index.get_level_values("currency")
    dummies = pd.get_dummies(currencies)

    if base_ccy not in dummies.columns:
        raise ValueError(f"base_ccy={base_ccy} not in currencies: {list(dummies.columns)}")

    for c in dummies.columns:
        if c == base_ccy:
            continue
        df[f"dollar_x_{c}"] = df["dollar"] * dummies[c].to_numpy()

    exog_vars = ["carry", "dollar"] + [f"dollar_x_{c}" for c in dummies.columns if c != base_ccy]

    model = PanelOLS(df["rx"], df[exog_vars], entity_effects=True)
    res = model.fit(cov_type="clustered", cluster_time=True)

    # --- recover total dollar beta per currency ---
    betas = {base_ccy: res.params["dollar"]}
    for c in dummies.columns:
        if c == base_ccy:
            continue
        betas[c] = res.params["dollar"] + res.params[f"dollar_x_{c}"]
    betas = pd.Series(betas).reindex(sorted(dummies.columns))

    return res, betas


def run_panel_regression_with_cds(fx_ret, carry, dollar, cds):
    ccy_cds = ["JPY", "MXN", "ZAR", "KRW", "SGD", "HKD", "INR", "BRL"]

    res_cds = stage1_panel_regression_cds(
        rx=fx_ret[ccy_cds],
        carry=carry[ccy_cds],
        dollar=dollar,
        cds=cds[
            ccy_cds
        ],  # make sure cds is aligned + in decimal (bps/10000) or your z-score
    )

    return res_cds


def run_insample_hedge_regression():
    data_dir = Path(__file__).parent.parent / "data"
    fx_data = data_dir / "fx_data"

    start_date = "2022-01-03"
    end_Date = "2024-12-31"

    # --- FX Excess Returns ---
    fx_ret = calculate_fx_excess_returns(data_dir, fx_data, start_date, end_Date)

    # --- Dollar ETF ---

    etf = pd.read_excel(data_dir / "dollar_etf.xlsx", parse_dates=["Dates"])
    etf.set_index("Dates", inplace=True)
    etf_ret = etf.pct_change()
    etf_ret.dropna(inplace=True)

    # --- Cut to In-Sample ---
    fx_ret = fx_ret.loc[pd.to_datetime(start_date): pd.to_datetime(end_Date)]
    etf = etf.loc[pd.to_datetime(start_date): pd.to_datetime(end_Date)]

    # --- Collect In-Sample Country-Specific Hedge Ratios ---

    results = []
    for col in fx_ret.columns:
        model = run_ols(etf_ret, fx_ret[col], add_const=False, verbose=False)
        results.append(
            {
                "fx": col,
                "beta": model.params.iloc[0],
                "std_err": model.bse.iloc[0],
                "t_stat": model.tvalues.iloc[0],
                "p_value": model.pvalues.iloc[0],
                "r2": model.rsquared,
            }
        )

    beta_table = pd.DataFrame(results).set_index("fx")
    beta_table.to_csv(data_dir / "is_hedge_ratios.csv")

    return beta_table