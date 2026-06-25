"""STEP 5: SARIMA baseline (Bab III)."""
import warnings, json
import numpy as np
import pandas as pd
import joblib
import pmdarima as pm
import matplotlib.pyplot as plt
from config import (DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS,
                    SARIMA_SEASONAL_PERIOD, SARIMA_MAX_P, SARIMA_MAX_Q)
warnings.filterwarnings("ignore")


def main():
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_PROCESSED / "train.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_PROCESSED / "val.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0, parse_dates=True)

    y_train = pd.concat([train["sales"], val["sales"]])   # train+val for final fit
    exog_cols = ["composite_index"]
    X_train = pd.concat([train[exog_cols], val[exog_cols]])
    X_test = test[exog_cols]

    m = min(SARIMA_SEASONAL_PERIOD, len(y_train) // 2)
    print(f"auto_arima (seasonal m={m})...")
    model = pm.auto_arima(
        y_train, X=X_train, seasonal=True, m=m,
        max_p=SARIMA_MAX_P, max_q=SARIMA_MAX_Q,
        stepwise=True, suppress_warnings=True, error_action="ignore", trace=True,
    )

    with open(RESULTS_TABLES / "sarima_summary.txt", "w") as f:
        f.write(str(model.summary()))

    preds = model.predict(n_periods=len(test), X=X_test)
    pd.DataFrame({"actual": test["sales"].values, "pred": np.asarray(preds)},
                 index=test.index).to_csv(RESULTS_TABLES / "sarima_predictions.csv")
    joblib.dump(model, RESULTS_MODELS / "sarima_model.pkl")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, test["sales"].values, "k-", label="Actual")
    ax.plot(test.index, np.asarray(preds), "k--", label="SARIMA")
    ax.legend(); ax.set_title("SARIMA - Test Forecast"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_6_sarima_diagnostics.png", dpi=150); plt.close()
    print(f"SARIMA order: {model.order} x {model.seasonal_order}")


if __name__ == "__main__":
    main()
