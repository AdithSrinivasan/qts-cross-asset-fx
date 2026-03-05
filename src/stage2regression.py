import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.load_data import prepare_bbg_data
from pathlib import Path
import matplotlib.pyplot as plt
from plotnine import *

DATA_DIR = Path(__file__).parent.parent / "data"
FX_DATA = DATA_DIR / "fx_data"

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"


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


def ols_regression(X_train, y_train, X_test, y_test, threshold_pct=0.1):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_predictions = lr.predict(X_train)
    _, strong_accuracies_train, train_threshold = evaluate_signal(
        train_predictions, y_train, top_pcts=[threshold_pct]
    )
    train_r2 = lr.score(X_train, y_train)
    predictions = lr.predict(X_test)
    df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    # print(f"R^2 Score: {lr.score(X_test, y_test)}")
    _, strong_accuracies_test, test_threshold = evaluate_signal(
        predictions, y_test, top_pcts=[threshold_pct]
    )
    return (
        predictions,
        lr,
        df[["Actual"]],
        train_r2,
        lr.score(X_test, y_test),
        strong_accuracies_train,
        strong_accuracies_test,
        train_threshold,
        test_threshold,
    )


def train_OLS(threshold_pct=0.1):
    countries, country_residuals = load_stage2_data()

    train_r2s = {}
    test_r2s = {}
    train_accuracies = {}
    test_accuracies = {}
    train_thresholds = {}
    test_thresholds = {}

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
        #
        #
        #
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

    return (
        train_r2s,
        test_r2s,
        train_accuracies,
        test_accuracies,
        train_thresholds,
        test_thresholds,
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


# if __name__ == "__main__":
#     # Load stage 1 residuals
#     stage1_residuals = load_stage1_residuals()
#     print("Loaded Stage 1 residuals...")
#     print(stage1_residuals.tail())
#     # Load stage 2 data
#     countries, country_residuals = load_stage2_data()
#     print("Loaded Stage 2 data...")
#     # Train OLS regression and evaluate
#     (
#         train_r2s,
#         test_r2s,
#         train_accuracies,
#         test_accuracies,
#         train_thresholds,
#         test_thresholds,
#     ) = train_OLS(country_residuals, countries, threshold_pct=0.1)
#     # Display results
#     display_r2s(train_r2s, test_r2s, model="OLS")
#     display_accuracies(
#         train_accuracies, threshold_pct=0.1, model="OLS", train_or_test="Training"
#     )
#     display_accuracies(
#         test_accuracies, threshold_pct=0.1, model="OLS", train_or_test="Test"
#     )
#     # Train Random Forest regression and evaluate
#     (
#         rf_train_r2s,
#         rf_test_r2s,
#         rf_train_accuracies,
#         rf_test_accuracies,
#         rf_train_thresholds,
#         rf_test_thresholds,
#         rf_test_preds,
#         rf_train_preds,
#     ) = train_random_forest(country_residuals, countries, threshold_pct=0.1)
#     # Display results
#     display_r2s(rf_train_r2s, rf_test_r2s, model="Random Forest")
#     display_accuracies(
#         rf_train_accuracies,
#         threshold_pct=0.1,
#         model="Random Forest",
#         train_or_test="Training",
#     )
#     display_accuracies(
#         rf_test_accuracies,
#         threshold_pct=0.1,
#         model="Random Forest",
#         train_or_test="Test",
#     )
