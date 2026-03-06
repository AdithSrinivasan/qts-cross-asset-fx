import pandas as pd
from pathlib import Path
from linearmodels.panel import PanelOLS
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from src.load_data import (
    prepare_fx_carry_data,
    calculate_fx_excess_returns,
    prepare_bbg_data,
)
import os
import numpy as np
from scipy import stats
import yfinance as yf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from src.load_data import prepare_bbg_data
import matplotlib.pyplot as plt
from plotnine import *

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"


def generate_thresholds(percentiles=0.9):
    df = pd.read_csv(DATA_DIR / "rf_train_predictions.csv", index_col="date")
    train_thresholds = {}
    for col in df.columns:
        threshold = df[col].abs().quantile(percentiles)
        train_thresholds[col] = threshold
    if percentiles == 0.9:
        output_path = DATA_DIR / "rf_thresholds.csv"
    elif percentiles == 0.8:
        output_path = DATA_DIR / "rf_exit_thresholds.csv"
    else:
        raise ValueError(
            "Unsupported percentiles value. Use 0.9 for entry thresholds or 0.8 for exit thresholds."
        )
    df_test = pd.read_csv(DATA_DIR / "rf_test_predictions.csv", index_col="date")
    test_thresholds = {}
    for col in df_test.columns:
        threshold = df_test[col].abs().quantile(percentiles)
        test_thresholds[col] = threshold

    thresholds_df = pd.DataFrame(
        {
            "Country": list(train_thresholds.keys()),
            "Train Threshold": list(train_thresholds.values()),
            "Test Threshold": list(test_thresholds.values()),
        }
    )
    thresholds_df.to_csv(output_path, index=True)


def load_stage1_residuals():
    stage1_res_path = DATA_DIR / "stage1_residuals.csv"
    if not stage1_res_path.exists():
        raise FileNotFoundError(
            f"Stage 1 residuals file not found at {stage1_res_path}. Please run stage 1 regression first."
        )
    res_df = pd.read_csv(stage1_res_path, parse_dates=["date"], index_col="date")
    return res_df


def load_stage2_data():
    eqy = pd.read_excel(DATA_DIR / "equity_indices.xlsx", parse_dates=["Dates"])
    eqy = prepare_bbg_data(eqy, START_DATE, END_DATE)

    spx = yf.download("^GSPC", start=START_DATE, end=END_DATE)

    eqy["SPX Index"] = spx["Close"]

    # print(eqy.head())
    eqy_returns = eqy.pct_change().fillna(method="ffill")

    mex = eqy_returns[["MEXBOL Index", "SPX Index"]]
    bra = eqy_returns[["IBOV Index", "SPX Index"]]
    saf = eqy_returns[["JALSH Index", "SPX Index"]]
    jpn = eqy_returns[["NKY Index", "SPX Index"]]
    aus = eqy_returns[["AS51 Index", "SPX Index"]]
    can = eqy_returns[["SPTSX Index", "SPX Index"]]
    gb = eqy_returns[["UKX Index", "SPX Index"]]
    hk = eqy_returns[["HSI Index", "SPX Index"]]
    ind = eqy_returns[["NIFTY Index", "SPX Index"]]
    kor = eqy_returns[["KOSPI Index", "SPX Index"]]
    nrw = eqy_returns[["OSEBX Index", "SPX Index"]]
    swd = eqy_returns[["OMX Index", "SPX Index"]]
    sng = eqy_returns[["STI Index", "SPX Index"]]
    sws = eqy_returns[["SMI Index", "SPX Index"]]
    nzw = eqy_returns[["NZ50SDE Index", "SPX Index"]]

    countries = {
        "Mexico": mex,
        "Brazil": bra,
        "South Africa": saf,
        "Japan": jpn,
        "Australia": aus,
        "Canada": can,
        "UK": gb,
        "Hong Kong": hk,
        "India": ind,
        "South Korea": kor,
        "Norway": nrw,
        "Sweden": swd,
        "Singapore": sng,
        "Switzerland": sws,
        "New Zealand": nzw,
    }
    for country, data in countries.items():
        data["Excess Return"] = data.iloc[:, 0] - data["SPX Index"]
        data["rolling 20 day volatility"] = (
            data["Excess Return"].rolling(window=20).std()
        )
        data["rolling 10 day volatility"] = (
            data["Excess Return"].rolling(window=10).std()
        )
        data["rolling 5 day volatility"] = data["Excess Return"].rolling(window=5).std()
        data["rolling 20 day mean"] = data["Excess Return"].rolling(window=20).mean()
        data["rolling 10 day mean"] = data["Excess Return"].rolling(window=10).mean()
        data["rolling 5 day mean"] = data["Excess Return"].rolling(window=5).mean()
        data["rolling 3 day mean"] = data["Excess Return"].rolling(window=3).mean()
    residuals = load_stage1_residuals()
    country_residuals = {}
    country_residuals["Australia"] = residuals["AUD"]

    country_residuals["Canada"] = residuals["CAD"]
    # print(country_residuals["Canada"].tail())
    country_residuals["Japan"] = residuals["JPY"]
    country_residuals["Mexico"] = residuals["MXN"]
    country_residuals["South Africa"] = residuals["ZAR"]
    country_residuals["UK"] = residuals["GBP"]
    country_residuals["Brazil"] = residuals["BRL"]
    country_residuals["Hong Kong"] = residuals["HKD"]
    country_residuals["India"] = residuals["INR"]
    country_residuals["South Korea"] = residuals["KRW"]
    country_residuals["Norway"] = residuals["NOK"]
    country_residuals["Sweden"] = residuals["SEK"]
    country_residuals["Singapore"] = residuals["SGD"]
    country_residuals["Switzerland"] = residuals["CHF"]
    country_residuals["New Zealand"] = residuals["NZD"]

    return countries, country_residuals


def evaluate_signal(predictions, y_test, top_pcts=[0.1]):
    predictions = pd.Series(predictions, index=y_test.index)
    y_test = pd.Series(y_test)
    directional_accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
    # print("Overall Directional Accuracy:", directional_accuracy)
    strong_accuracies = []
    threshold = 0
    for top_pct in top_pcts:
        threshold = predictions.abs().quantile(1 - top_pct)
        # print(f"Threshold for top {int(top_pct*100)}% signals: {threshold:.4f}")
        strong_mask = predictions.abs() >= threshold
        strong_directional_accuracy = np.mean(
            np.sign(predictions[strong_mask]) == np.sign(y_test[strong_mask])
        )
        # print(f"Top {int(top_pct*100)}% Strongest Signals:")
        # print(f"Signal Threshold: {threshold:.4f}")
        # print("Directional Accuracy:", strong_directional_accuracy)
        strong_accuracies.append((top_pct, strong_directional_accuracy))
    return (
        {
            "overall_direction": directional_accuracy,
            "strong_direction": strong_directional_accuracy,
        },
        strong_accuracies,
        threshold,
    )


def calculate_r2(predictions, y_true):
    ss_res = np.sum((y_true - predictions) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score


def ols_regression(X_train, y_train, X_test, y_test, threshold_pct=0.1):
    lr = sm.OLS(y_train, sm.add_constant(X_train))
    lr = lr.fit()
    pvalues = lr.pvalues[1:]  # Exclude intercept
    train_predictions = lr.predict(sm.add_constant(X_train))
    _, strong_accuracies_train, train_threshold = evaluate_signal(
        train_predictions, y_train, top_pcts=[threshold_pct]
    )
    train_r2 = lr.rsquared
    predictions = lr.predict(sm.add_constant(X_test))

    df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    # print(f"R^2 Score: {lr.score(X_test, y_test)}")
    _, strong_accuracies_test, test_threshold = evaluate_signal(
        predictions, y_test, top_pcts=[threshold_pct]
    )
    test_r2 = calculate_r2(predictions, y_test)
    return (
        predictions,
        lr,
        df[["Actual"]],
        train_r2,
        test_r2,
        strong_accuracies_train,
        strong_accuracies_test,
        train_threshold,
        test_threshold,
        pvalues,
    )


def train_OLS(threshold_pct=0.1):
    countries, country_residuals = load_stage2_data()

    train_r2s = {}
    test_r2s = {}
    train_accuracies = {}
    test_accuracies = {}
    train_thresholds = {}
    test_thresholds = {}
    pvalues = {}

    for country, data in countries.items():
        if country not in country_residuals:
            # print(f"Residuals for {country} not found. Skipping regression.")
            continue
        # print(f"\nRunning regression for {country}...")
        # print(data.head())
        X_train = countries[country]["2022-01-01":"2024-09-30"]
        y_train = (
            country_residuals[country]["2022-01-01":"2024-09-30"].shift(-1).dropna()
        )  # Shift to align with next day's return
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_train.index = pd.to_datetime(X_train.index)
        y_train.index = pd.to_datetime(y_train.index)
        df_train = pd.concat([X_train, y_train], axis=1).dropna()
        df_train.drop(
            columns=["SPX Index"], inplace=True
        )  # Drop SPX Index from features

        X_test = countries[country]["2025-01-01":"2026-01-01"]
        y_test = (
            country_residuals[country]["2025-01-01":"2026-01-01"].shift(-1).dropna()
        )  # Shift to align with next day's return
        X_test = X_test.copy()
        y_test = y_test.copy()

        # print(y_test.head())
        X_test.index = pd.to_datetime(X_test.index)
        y_test.index = pd.to_datetime(y_test.index)
        df_test = pd.concat([X_test, y_test], axis=1).dropna()

        df_test.drop(
            columns=["SPX Index"], inplace=True
        )  # Drop SPX Index from features
        (
            model_preds,
            model,
            ys,
            trains_r2,
            test_r2,
            strong_acc_train,
            strong_acc_test,
            train_threshold,
            test_threshold,
            pvalue,
        ) = ols_regression(
            df_train[["Excess Return"]],
            df_train.iloc[:, -1],
            df_test[["Excess Return"]],
            df_test.iloc[:, -1],
            threshold_pct=threshold_pct,
        )
        train_r2s[country] = trains_r2
        test_r2s[country] = test_r2
        train_accuracies[country] = strong_acc_train
        test_accuracies[country] = strong_acc_test
        train_thresholds[country] = train_threshold
        test_thresholds[country] = test_threshold
        pvalues[country] = pvalue
    pvalues = pd.DataFrame(pvalues).T
    pvalues["Excess Returns p-value"] = pvalues["Excess Return"]
    pvalues = pvalues[["Excess Returns p-value"]]
    return (
        train_r2s,
        test_r2s,
        train_accuracies,
        test_accuracies,
        train_thresholds,
        test_thresholds,
        pvalues,
    )


def display_r2s(train_r2s, test_r2s, model="Random Forest"):

    df = pd.DataFrame(
        {
            "Country": list(train_r2s.keys()),
            "Train": list(train_r2s.values()),
            "Test": [test_r2s[c] for c in train_r2s.keys()],
        }
    )

    df = df.melt(id_vars="Country", var_name="Dataset", value_name="R2")

    return (
        ggplot(df, aes(x="Country", y="R2", fill="Dataset"))
        + geom_col(position="identity", alpha=0.5)
        + labs(title=f"{model} R² Scores by Country", y="R² Score", x="Country")
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + theme(figure_size=(10, 6))
    )


def display_accuracies(
    accuracies, threshold_pct=0.1, model="Random Forest", train_or_test="Training"
):

    df = pd.DataFrame(
        {
            "Country": list(accuracies.keys()),
            "Accuracy": [v[0][1] for v in accuracies.values()],
        }
    )

    return (
        ggplot(df, aes(x="Country", y="Accuracy"))
        + geom_col(color="steelblue", fill="steelblue", alpha=0.7)
        + geom_hline(yintercept=0.5, linetype="dashed")
        + labs(
            title=f"Top {int(threshold_pct*100)}% Strongest Signal Directional Accuracy - {train_or_test} Data ({model})",
            x="Country",
            y="Directional Accuracy",
        )
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + theme(figure_size=(10, 6))
    )


def random_forest_model(X_train, y_train, X_test, y_test, threshold_pct=0.1):
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=4, min_samples_leaf=10, random_state=42, n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # print("\nImportance-Based Feature Selection:")
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )

    top_features = importances.index[: int(len(importances) * 0.5)]

    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]

    rf_reduced = RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=15, random_state=42, n_jobs=-1
    )

    rf_reduced.fit(X_train_reduced, y_train)
    train_predictions = rf_reduced.predict(X_train_reduced)
    train_r2 = rf_reduced.score(X_train_reduced, y_train)
    # print("\nTraining Data:")
    # print("Train R² (Reduced):", train_r2)
    dic, strong_accuracies_train, train_threshold = evaluate_signal(
        train_predictions, y_train, [threshold_pct]
    )
    # print()
    # print("\nTesting Data:")
    predictions_reduced = rf_reduced.predict(X_test_reduced)
    test_r2 = rf_reduced.score(X_test_reduced, y_test)
    # print("Reduced Test R²:", test_r2)
    dic, strong_accuracies_test, test_threshold = evaluate_signal(
        predictions_reduced, y_test, [threshold_pct]
    )

    return (
        predictions_reduced,
        train_predictions,
        rf_reduced,
        y_test,
        y_train,
        train_r2,
        test_r2,
        strong_accuracies_train,
        strong_accuracies_test,
        train_threshold,
        test_threshold,
    )


def train_random_forest(threshold_pct=0.1):
    countries, country_residuals = load_stage2_data()
    train_r2s = {}
    test_r2s = {}
    train_accuracies = {}
    test_accuracies = {}
    train_thresholds = {}
    test_thresholds = {}
    test_preds = pd.DataFrame()
    train_preds = pd.DataFrame()

    for country, data in countries.items():
        if country not in country_residuals:
            # print(f"\nResiduals for {country} not found. Skipping regression.")
            continue
        X_train = countries[country]["2022-01-01":"2024-09-30"]
        y_train = (
            country_residuals[country]["2022-01-01":"2024-09-30"].shift(-1).dropna()
        )  # Shift to align with next day's return
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_train.index = pd.to_datetime(X_train.index)
        y_train.index = pd.to_datetime(y_train.index)
        df_train = pd.concat([X_train, y_train], axis=1).dropna()
        df_train.drop(
            columns=["SPX Index"], inplace=True
        )  # Drop SPX Index from features

        X_test = countries[country]["2025-01-01":"2026-01-01"]
        y_test = (
            country_residuals[country]["2025-01-01":"2026-01-01"].shift(-1).dropna()
        )  # Shift to align with next day's return
        X_test = X_test.copy()
        y_test = y_test.copy()
        X_test.index = pd.to_datetime(X_test.index)
        y_test.index = pd.to_datetime(y_test.index)
        df_test = pd.concat([X_test, y_test], axis=1).dropna()
        df_test.drop(
            columns=["SPX Index"], inplace=True
        )  # Drop SPX Index from features

        # print(f"\nRandom Forest Regression Results for {country}:")
        (
            rf_test_predictions,
            rf_train_predictions,
            rf_model_instance,
            y_test,
            y_train,
            train_r2,
            test_r2,
            strong_acc_train,
            strong_acc_test,
            train_threshold,
            test_threshold,
        ) = random_forest_model(
            df_train.iloc[:, :-1],
            df_train.iloc[:, -1],
            df_test.iloc[:, :-1],
            df_test.iloc[:, -1],
            threshold_pct=threshold_pct,
        )
        if country == "Mexico":
            test_preds.index = y_test.index
            train_preds.index = y_train.index
        test_preds[country] = rf_test_predictions
        train_preds[country] = rf_train_predictions
        train_r2s[country] = train_r2
        test_r2s[country] = test_r2
        train_accuracies[country] = strong_acc_train
        test_accuracies[country] = strong_acc_test
        train_thresholds[country] = train_threshold
        test_thresholds[country] = test_threshold
    return (
        train_r2s,
        test_r2s,
        train_accuracies,
        test_accuracies,
        train_thresholds,
        test_thresholds,
        test_preds,
        train_preds,
    )


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
    y_clean = combined.iloc[:, 0]
    x_clean = combined.iloc[:, 1:]
    X = add_constant(x_clean) if add_const else x_clean
    model = OLS(y_clean, X).fit()
    if verbose:
        print(model.summary())
    return model


def stage1_panel_regression(rx, carry, dollar):
    # --- long format ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")

    df = df.merge(
        dollar.rename("dollar"), left_on="date", right_index=True, how="inner"
    )

    df = df.set_index(["currency", "date"]).sort_index()

    exog_vars = ["carry", "dollar"]

    exog = df[exog_vars]

    model = PanelOLS(df["rx"], exog, entity_effects=True)

    res = model.fit(cov_type="clustered", cluster_time=True)

    return res


def stage1_panel_regression_cds(rx, carry, dollar, cds):
    # --- long format ---
    rx_long = rx.stack().rename("rx").reset_index()
    rx_long.columns = ["date", "currency", "rx"]

    carry_long = carry.stack().rename("carry").reset_index()
    carry_long.columns = ["date", "currency", "carry"]

    df = rx_long.merge(carry_long, on=["date", "currency"], how="inner")
    df = df.merge(
        dollar.rename("dollar"), left_on="date", right_index=True, how="inner"
    )

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
    df = df.merge(
        dollar.rename("dollar"), left_on="date", right_index=True, how="inner"
    )
    df = df.set_index(["currency", "date"]).sort_index()

    # --- currency-specific dollar interaction terms ---
    currencies = df.index.get_level_values("currency")
    dummies = pd.get_dummies(currencies)

    if base_ccy not in dummies.columns:
        raise ValueError(
            f"base_ccy={base_ccy} not in currencies: {list(dummies.columns)}"
        )

    for c in dummies.columns:
        if c == base_ccy:
            continue
        df[f"dollar_x_{c}"] = df["dollar"] * dummies[c].to_numpy()

    exog_vars = ["carry", "dollar"] + [
        f"dollar_x_{c}" for c in dummies.columns if c != base_ccy
    ]

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
    fx_ret = fx_ret.loc[pd.to_datetime(start_date) : pd.to_datetime(end_Date)]
    etf = etf.loc[pd.to_datetime(start_date) : pd.to_datetime(end_Date)]

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
